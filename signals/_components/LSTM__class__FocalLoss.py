import logging
import sys
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent.parent)) if str(Path(__file__).parent.parent.parent) not in sys.path else None
from utilities._logger import setup_logging

logger = setup_logging(module_name="LSTM__class__FocalLoss", log_level=logging.DEBUG)

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance in classification tasks.
    
    Focal Loss addresses class imbalance by down-weighting easy examples and focusing
    on hard negatives. The loss is computed as: FL(pt) = -α(1-pt)^γ * log(pt)
    
    Args:
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter to down-weight easy examples (default: 2.0)
        reduction: Specifies the reduction to apply to the output
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ) -> None:
        super().__init__()
        
        # Validate reduction parameter
        valid_reductions = ['mean', 'sum', 'none']
        if reduction not in valid_reductions:
            raise ValueError(f"Invalid reduction '{reduction}'. Must be one of {valid_reductions}")
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits of shape (N, C) where N is batch size, C is number of classes
            targets: Ground truth labels of shape (N,) with class indices
            
        Returns:
            Computed focal loss tensor
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss