"""
This module contains the ModelTrainer class, which is responsible for
training the PyTorch models for signal generation.
"""

import sys
from pathlib import Path
import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utilities._logger import setup_logging

logger = setup_logging(module_name="ModelTrainer", log_level=logging.INFO)

class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric has stopped improving.
    """
    def __init__(self, patience: int = 7, min_delta: float = 0, verbose: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                
class ModelTrainer:
    """
    A class to handle the training and evaluation of PyTorch models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        use_amp: bool = False,
    ):
        """
        Initializes the ModelTrainer.

        Args:
            model: The PyTorch model to train.
            optimizer: The optimizer.
            criterion: The loss function.
            device: The device to train on (cpu or cuda).
            use_amp: Whether to use Automatic Mixed Precision.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=self.use_amp)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.early_stopper = None

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, use_early_stopping: bool = True):
        """
        Runs the full training loop for the specified number of epochs.
        """
        if use_early_stopping:
            self.early_stopper = EarlyStopping(patience=10, min_delta=1e-4)

        for epoch in range(epochs):
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            end_time = time.time()

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Time: {end_time - start_time:.2f}s"
            )

            self.scheduler.step(val_loss)

            if use_early_stopping and self.early_stopper:
                self.early_stopper(val_loss)
                if self.early_stopper.early_stop:
                    logger.info("Early stopping triggered.")
                    break

    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Runs a single training epoch.
        """
        self.model.train()
        total_loss = 0
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

            self.scaler.scale(loss).backward()  # type: ignore
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluates the model on a dataset.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        return total_loss / len(data_loader) 