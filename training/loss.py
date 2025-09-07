import torch.nn as nn
import torch

class GPT2Loss(nn.Module):
    """Cross Entropy Loss Function wrapper function

        Cross Entropy is a generalized, expanded form of Log-Loss 
        (two-class vs. multi-class classification tasks)
        We can assess an LLM’s prediction confidence by examining 
        its output for cross entropy, revealing how certain/confident 
        it was in making those predictions.

        attributes:
            ignore_index (int) : Padding token index to be ignored in calculating loss
            label_smoothing(float): label smoothing distribute a small portion of the probability mass
            from the correct class to other classes. It prevents LLMs from becoming overconfident, 
            improves model generalization, and enhances robustness by making the model less certain 
            about its predictions and more resilient to mislabeled or noisy data.

        methods:
            forward(self , x:torch.Tensor)->torch.Tensor:
                -Forward method for calculating loss

    """
    def __init__(self,ignore_index:int,label_smoothing:float=0.05) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index,label_smoothing= label_smoothing)
    

    def forward(self,logits:torch.Tensor,labels:torch.Tensor)->torch.Tensor:
        """Forward method implementation for loss function
            Args:
                logits(torch.Tensor): output logits from model
                labels(torch.Tensor): actual label

            returns:
                torch.Tensor : Loss for current prediction        
        """
        return self.loss_fn(logits.flatten(0,1),labels.flatten())
