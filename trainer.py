import torch
from training.loss import GPT2Loss
from gpt2.model import GPT2Model
from torch.utils.data import DataLoader   
from typing import Tuple
import os


class GPT2Trainer:

    def __init__(self,
                 model:GPT2Model,
                 train_dataloader:DataLoader[Tuple[torch.Tensor, torch.Tensor]]
                ,val_dataloader:DataLoader[Tuple[torch.Tensor, torch.Tensor]],
                loss_fn:GPT2Loss,
                optimizer:torch.optim.Optimizer,
                device: torch.device |str,
                epochs:int,
                log_interval:int = 100,
                checkpoint_dir:str = "./experiments/checkpoints") -> None:
        
        """Initialize GPT2Trainer.

            Args:
                model: GPT2Model instance.
                train_dataloader: DataLoader yielding (input_ids, target_ids) for training.
                val_dataloader: DataLoader for validation, optional.
                loss_fn: Loss function (GPT2Loss).
                optimizer: PyTorch optimizer.
                device: torch.device or str ('cpu'/'cuda').
                epochs: Number of training epochs.
                log_interval: Steps between logging training stats.
                checkpoint_dir: Directory to save checkpoints.
        """
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model = model.to(self.device)

    def train_epoch(self)->float:
        """Run one training epoch.

        Returns:
            float: return average training loss for the epoch
        
        """
        
        if len(self.train_dataloader) == 0:
            raise ValueError("Train Dataloader is empty")
    
        self.model.train()
        total_loss = 0.0
        batch_loss = []
        for batch_no, (input_batch,target_batch) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            input_batch ,target_batch = input_batch.to(self.device),target_batch.to(self.device)
            logits = self.model(input_batch)
            loss = self.loss_fn(logits,target_batch)
            batch_loss.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if batch_no % self.log_interval == 0:
                avg_so_far = total_loss / (batch_no + 1)
                print(f"Step {batch_no:06d}: Train loss {avg_so_far:.3f}, Last batch {batch_loss[-1]:.3f}")

        avg_loss = total_loss / len(self.train_dataloader) 
        return avg_loss

    def eval_epoch(self)->float:
        """Run One validation epoch


        Returns:
            float: return average validation loss for the epoch
        
        """
        
        if len(self.val_dataloader) == 0:
            raise ValueError("validation dataloader is empty")
        self.model.eval()
        total_loss = 0.0
        for input_batch , target_batch in self.val_dataloader:
            input_batch ,target_batch = input_batch.to(self.device),target_batch.to(self.device)
            with torch.no_grad():
                logits = self.model(input_batch)
                loss = self.loss_fn(logits,target_batch)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        return avg_loss

    def save_checkpoint(self,checkpoint:dict[str, torch.Tensor],path:str|None=None)->None:
        """Save model and optimizer state to a checkpoint.
        
        Args:
            checkpoint(object): checkpoint object like model state dic, optimizer state dic etc
            path(str|None): path where checkpoint need to save (optional)
        """
        
        if path is None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path =  os.path.join(self.checkpoint_dir, 'model_checkpoint.pth')
       
        torch.save(checkpoint, path)


    def load_checkpoint(self,path:str)->None:
        """Load model and optimizer state from a checkpoint
        
        Args:
            path(str): Checkpoint path
        
        Note: checkpoint should have state_dict and optimizer.
        """
        if not os.path.exists(path):
            raise FileNotFoundError("Path not exist. please check")
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(e)

   
