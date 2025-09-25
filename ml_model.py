"""
Machine Learning Model for Online Convex Optimization
Implements LSTM-based predictor for action sequences
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class LSTMOCOPredictor(nn.Module):
    """
    LSTM-based predictor for Online Convex Optimization
    
    Input: historical actions and contexts
    Output: predicted next action
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2, 
                 output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize LSTM predictor
        
        Args:
            input_dim: dimension of input features (action + context)
            hidden_dim: LSTM hidden dimension
            num_layers: number of LSTM layers
            output_dim: output dimension (1 for single action)
            dropout: dropout rate
        """
        super(LSTMOCOPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            predicted actions of shape (batch_size, seq_len, output_dim)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output layer
        output = self.fc(lstm_out)
        
        return output


class MLOCOTrainer:
    """
    Trainer for ML-based OCO predictor
    """
    
    def __init__(self, model: nn.Module, m: float = 5.0, learning_rate: float = 0.001):
        """
        Initialize trainer
        
        Args:
            model: LSTM predictor model
            m: hitting cost parameter
            learning_rate: learning rate for optimizer
        """
        self.model = model
        self.m = m
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = self.oco_loss
        
    def oco_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 prev_actions: torch.Tensor = None) -> torch.Tensor:
        """
        Compute OCO loss: m/2 * ||x_t - y_t||² + 1/2 * ||x_t - x_{t-1}||²
        
        Args:
            predictions: predicted actions x_t
            targets: target contexts y_t
            prev_actions: previous actions x_{t-1}
            
        Returns:
            OCO loss
        """
        # Hitting cost: m/2 * ||x_t - y_t||²
        hitting_cost = (self.m / 2) * torch.mean((predictions - targets) ** 2)
        
        # Switching cost: 1/2 * ||x_t - x_{t-1}||²
        if prev_actions is not None:
            switching_cost = (1 / 2) * torch.mean((predictions - prev_actions) ** 2)
        else:
            switching_cost = torch.tensor(0.0, device=predictions.device)
            
        return hitting_cost + switching_cost
    
    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: training data loader
            
        Returns:
            average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            inputs, targets, prev_actions = batch
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(inputs)
            
            # Compute loss
            loss = self.criterion(predictions, targets, prev_actions)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def evaluate(self, test_loader) -> Tuple[float, List[float]]:
        """
        Evaluate model on test set
        
        Args:
            test_loader: test data loader
            
        Returns:
            tuple of (average_loss, predictions)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets, prev_actions = batch
                
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets, prev_actions)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy().flatten())
                num_batches += 1
                
        return total_loss / num_batches, all_predictions


class MLOCOPredictor:
    """
    High-level interface for ML-based OCO prediction
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2,
                 m: float = 5.0, learning_rate: float = 0.001):
        """
        Initialize ML predictor
        
        Args:
            input_dim: input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: number of LSTM layers
            m: hitting cost parameter
            learning_rate: learning rate
        """
        self.model = LSTMOCOPredictor(input_dim, hidden_dim, num_layers)
        self.trainer = MLOCOTrainer(self.model, m, learning_rate)
        self.m = m
        
    def train(self, train_loader, num_epochs: int = 100, verbose: bool = True):
        """
        Train the model
        
        Args:
            train_loader: training data loader
            num_epochs: number of training epochs
            verbose: whether to print training progress
        """
        train_losses = []
        
        for epoch in range(num_epochs):
            loss = self.trainer.train_epoch(train_loader)
            train_losses.append(loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.6f}")
                
        return train_losses
    
    def predict(self, contexts: List[float], prev_actions: List[float] = None) -> Tuple[List[float], float]:
        """
        Predict actions for a sequence of contexts
        
        Args:
            contexts: list of contexts y_1, y_2, ..., y_T
            prev_actions: list of previous actions (None for first prediction)
            
        Returns:
            tuple of (predicted_actions, total_cost)
        """
        self.model.eval()
        actions = []
        total_cost = 0.0
        
        with torch.no_grad():
            for t, context in enumerate(contexts):
                if t == 0:
                    # First action: use simple prediction or context
                    action = context
                else:
                    # Prepare input features [prev_action, context]
                    prev_action = actions[-1] if actions else 0.0
                    input_features = torch.tensor([[prev_action, context]], dtype=torch.float32)
                    
                    # Predict next action
                    prediction = self.model(input_features)
                    action = prediction.item()
                
                actions.append(action)
                
                # Compute cost
                hitting_cost = (self.m / 2) * (action - context) ** 2
                switching_cost = (1 / 2) * (action - actions[-2]) ** 2 if len(actions) > 1 else 0.0
                total_cost += hitting_cost + switching_cost
                
        return actions, total_cost


def test_ml_predictor():
    """Test the ML predictor with a simple example"""
    # Create simple test data
    contexts = [0.2, 0.5, 0.8, 0.3, 0.7]
    
    predictor = MLOCOPredictor()
    actions, total_cost = predictor.predict(contexts)
    
    print("ML Predictor Test Results:")
    print(f"Contexts: {contexts}")
    print(f"Actions: {[f'{a:.3f}' for a in actions]}")
    print(f"Total Cost: {total_cost:.3f}")
    
    return actions, total_cost


if __name__ == "__main__":
    test_ml_predictor()
