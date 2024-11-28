from torch.utils.data import Dataset,DataLoader
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from transformer.model import TransformerDecoder
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler #Package for mixed precision in cuda computations

from typing import Dict
from pathlib import Path
import logging
from tqdm import tqdm
import wandb #Visualisations

#Custom Dataset
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length):
        """
        Args:
            text (str): Raw text data.
            tokenizer: Tokenizer to convert text to tokens.
            seq_length (int): Length of each sequence.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize the text and convert to input IDs
        self.tokens = tokenizer.encode(text,return_tensors="pt",add_special_tokens=False)[0]
        
        # Chunk tokens into sequences of fixed length
        self.num_sequences = self.tokens.size(-1) // seq_length
        self.tokens = self.tokens[: self.num_sequences * seq_length]  # Trim extra tokens

    def __len__(self):
        # Number of sequences
        return self.num_sequences

    def __getitem__(self, idx):
        # Return a sequence of fixed length
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        sequence = torch.cat((torch.tensor((self.tokenizer.bos_token_id,)),self.tokens[start_idx:end_idx]),dim=-1)
        sequence = torch.tensor(sequence, dtype=torch.long)
        return sequence[:-1],sequence[1:]
    
class CustomLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lrate = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5, 
            self.step_num * (self.warmup_steps ** -1.5)     
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lrate

class Trainer:
    def __init__(self,model:nn.Module,train_loader:DataLoader,val_loader:DataLoader,tokenizer:GPT2Tokenizer,config:dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9,0.98)
        )

        # # Define the learning rate scheduler
        self.scheduler = CustomLRScheduler(self.optimizer,model_config['d_model'],train_config['warmup_steps'])

        #Loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_idx) 

        #Mixed precision
        self.scaler = GradScaler()

        #Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Initialize logging ???
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def train_epoch(self,epoch:int):
        """
        Train for one epoch
        """
        self.model.train() #Set to train
        total_loss = 0
        progress_bar = tqdm(self.train_loader,desc=f'Epoch {epoch}')

        for batch_idx,batch in enumerate(progress_bar):
            #Move batch to device
            src_seq,tgt_seq = batch
            src_seq,tgt_seq = src_seq.to(self.device),tgt_seq.to(self.device)

            #Mixed precision training
            with autocast(self.device.type):
                outputs = self.model(src_seq,None) #logits of shape (batch_size,max_len,vocab_size)
                loss = self.loss_fn(outputs.view(train_config['batch_size']*model_config['max_len'],-1).softmax(dim=-1),tgt_seq.flatten())

            #Scale loss and compute gadients
            self.scaler.scale(loss).backward()

            #Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)

            #Update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            #Zero gradients
            self.optimizer.zero_grad()

            #update total loss
            total_loss+=loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss / (batch_idx + 1):.4f}",
                'lr': f"{self.scheduler.optimizer.param_groups[0]['lr']:.2e}"
            })

            #Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'train_loss':loss.item(),
                    'learning_rate':self.scheduler.optimizer.param_groups[0]['lr'],
                })

        # Compute epoch averages
        num_batches = len(self.train_loader)
        epoch_loss = total_loss / num_batches

        return epoch_loss,None
    
    def train(self):
        """Full training loop"""
        for epoch in range(train_config['num_epochs']):
            self.logger.info(f"\nStarting epoch {epoch + 1}")
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            #val_loss, val_metrics = self.validate()
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                #'val_loss': val_loss,
                #**{f"train_{k}": v for k, v in train_metrics.items()},
                #**{f"val_{k}": v for k, v in val_metrics.items()}
            }
            
            self.logger.info(
                f"Epoch {epoch + 1} - "
                f"Train loss: {train_loss:.4f} - "
                #f"Val loss: {val_loss:.4f} - "
               # f"Val BLEU: {val_metrics['bleu']:.2f}"
            )
            
            if wandb.run is not None:
                wandb.log(metrics)
            
            # Save checkpoint if best so far
            #if val_metrics['bleu'] > best_bleu:
                #best_bleu = val_metrics['bleu']
                #self.save_checkpoint(epoch, metrics)




if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # Use GPT-2 tokenizer (or another tokenizer)
    tokenizer.pad_token = tokenizer.eos_token # Add a padding token (if required for batching)

    # __ CONFIG ___
    base_model_config = {
        'd_model' : 256,
        'd_k' : 32,
        'd_v' : 32,
        'd_ff' : 1024,
        'n_layers' : 4,
        'n_head' : 8,
        'max_len' : 128,
        'vocab_size' : tokenizer.vocab_size,
        'pad_idx':tokenizer.pad_token_id,
        'dropout' : 0.1,
    }

    mini_model_config = {
        'd_model' : 64,
        'd_k' : 8,
        'd_v' : 8,
        'd_ff' : 256,
        'n_layers' : 2,
        'n_head' : 2,
        'max_len' : 32,
        'vocab_size' : tokenizer.vocab_size,
        'pad_idx':tokenizer.pad_token_id,
        'dropout' : 0.1,
    }

    model_config = mini_model_config

    train_config = {
        'learning_rate':1e-9,
        'weight_decay':1e-5,
        'checkpoint_dir':".",
        'num_epochs': 50,
        'warmup_steps': 1000,
        'batch_size':64,
        'train_per':0.9
    }

    # Path to the .txt file
    file_path = "Attention/tiny_shakespeare.txt"

    # Open the file and read its contents as a single string
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    #Train/Val split
    text_lengh = len(text)
    sep = int(text_lengh*train_config['train_per'])
    train_text = text[:sep]
    val_text = text[sep:]

    #Create a dataset
    train_dataset = TextDataset(train_text, tokenizer, model_config['max_len'])
    val_dataset = TextDataset(val_text, tokenizer, model_config['max_len'])

    # Create the DataLoader and Explicitly specify pin_memory for CUDA for faster data transfer
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config['batch_size'], 
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available()  # This helps with CUDA performance
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_config['batch_size'], 
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )

    #create a model
    model = TransformerDecoder(**model_config)

    #create trainer
    trainer = Trainer(model,train_loader,val_loader,tokenizer,train_config)

    #Train
    trainer.train()