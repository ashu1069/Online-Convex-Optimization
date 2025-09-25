"""
Data processing utilities for vision-based controller tuning
Handles loading, preprocessing, and sequence generation for scene signals
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class AIDataset(Dataset):
    """
    Dataset class for AI workload data
    """
    
    def __init__(self, sequences: List[np.ndarray], sequence_length: int = 24):
        """
        Initialize dataset
        
        Args:
            sequences: list of time sequences
            sequence_length: length of each sequence
        """
        self.sequences = sequences
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Input: previous actions and current context
        # For simplicity, we'll use context as both input and target
        inputs = []
        targets = []
        prev_actions = []
        
        for t in range(1, len(sequence)):
            # Input features: [prev_action, current_context]
            prev_action = sequence[t-1] if t > 0 else 0.0
            current_context = sequence[t]
            
            inputs.append([prev_action, current_context])
            targets.append(current_context)
            prev_actions.append(prev_action)
        
        # Convert to tensors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        prev_actions = torch.tensor(prev_actions, dtype=torch.float32).unsqueeze(-1)
        
        return inputs, targets, prev_actions


class DataProcessor:
    """
    Data processor for AI workload dataset
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize data processor
        
        Args:
            data_path: path to AI workload CSV file
        """
        self.data_path = data_path
        self.raw_data = None
        self.normalized_data = None
        self.sequences = []
        self.train_sequences = []
        self.test_sequences = []
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load AI workload data from CSV
        
        Args:
            data_path: path to CSV file
            
        Returns:
            loaded dataframe
        """
        if data_path is None:
            data_path = self.data_path
            
        if data_path is None:
            # Create synthetic data for testing if no file provided
            print("No data file provided, creating synthetic vision data...")
            return self._create_synthetic_data()
        
        try:
            self.raw_data = pd.read_csv(data_path)
            print(f"Loaded vision data with shape: {self.raw_data.shape}")
            return self.raw_data
        except FileNotFoundError:
            print(f"File {data_path} not found, creating synthetic vision data...")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """
        Create synthetic AI workload data for testing
        
        Returns:
            synthetic dataframe
        """
        # Generate synthetic scene signals for vision (exposure target proxies)
        # 19 days of hourly data (19 * 24 = 456 frames for simplicity)
        hours = np.arange(0, 456)

        # Scene brightness (lux proxy) with diurnal and weekly trends plus noise
        base_brightness = 0.6  # normalized baseline
        daily_cycle = 0.3 * np.clip(np.sin(2 * np.pi * hours / 24) + 0.2, 0, 1)
        weekly_cycle = 0.1 * (np.sin(2 * np.pi * hours / (24 * 7)) + 1) / 2
        transients = 0.15 * np.random.rand(len(hours))  # sudden lighting changes
        brightness = base_brightness * (0.5 + daily_cycle) + weekly_cycle + transients

        # Feature density proxy (edges/texture), anticorrelated at night, noisy
        feature_density = 0.7 - 0.4 * daily_cycle + 0.15 * np.random.randn(len(hours))

        # Clamp to positive range before normalization later
        brightness = np.maximum(brightness, 1e-3)
        feature_density = np.maximum(feature_density, 1e-3)

        df = pd.DataFrame({
            'time_index': hours,
            'scene_brightness': brightness,
            'feature_density': feature_density
        })
        
        return df
    
    def normalize_data(self, data: pd.DataFrame = None) -> np.ndarray:
        """
        Normalize data to [0, 1] range
        
        Args:
            data: dataframe to normalize
            
        Returns:
            normalized data array
        """
        if data is None:
            data = self.raw_data
            
        if data is None:
            raise ValueError("No data to normalize")
        
        # Prefer scene_brightness; fall back to first numeric column
        signal_col = 'scene_brightness'
        if signal_col not in data.columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            signal_col = numeric_cols[0]

        signal = data[signal_col].values

        # Normalize to [0, 1]
        min_val = np.min(signal)
        max_val = np.max(signal)
        self.normalized_data = (signal - min_val) / (max_val - min_val + 1e-12)
        
        print(f"Normalized scene signal range: [{np.min(self.normalized_data):.3f}, {np.max(self.normalized_data):.3f}]")
        
        return self.normalized_data
    
    def generate_sequences(self, window_size: int = 24, step_size: int = 2) -> List[np.ndarray]:
        """
        Generate time sequences using sliding window
        
        Args:
            window_size: size of each sequence (24 hours)
            step_size: step size for sliding window (2 hours)
            
        Returns:
            list of sequences
        """
        if self.normalized_data is None:
            raise ValueError("Data must be normalized first")
        
        sequences = []
        
        for start_idx in range(0, len(self.normalized_data) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            sequence = self.normalized_data[start_idx:end_idx]
            sequences.append(sequence)
        
        self.sequences = sequences
        print(f"Generated {len(sequences)} sequences of length {window_size}")
        
        return sequences
    
    def split_data(self, train_ratio: float = 0.8) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Split sequences into training and testing sets
        
        Args:
            train_ratio: ratio of data for training
            
        Returns:
            tuple of (train_sequences, test_sequences)
        """
        if not self.sequences:
            raise ValueError("No sequences generated")
        
        split_idx = int(len(self.sequences) * train_ratio)
        
        self.train_sequences = self.sequences[:split_idx]
        self.test_sequences = self.sequences[split_idx:]
        
        print(f"Split data: {len(self.train_sequences)} train, {len(self.test_sequences)} test sequences")
        
        return self.train_sequences, self.test_sequences
    
    def create_data_loaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch data loaders
        
        Args:
            batch_size: batch size for data loaders
            
        Returns:
            tuple of (train_loader, test_loader)
        """
        if not self.train_sequences or not self.test_sequences:
            raise ValueError("Data must be split first")
        
        train_dataset = AIDataset(self.train_sequences)
        test_dataset = AIDataset(self.test_sequences)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def visualize_data(self, save_path: str = None):
        """
        Visualize the processed data
        
        Args:
            save_path: path to save the plot
        """
        if self.raw_data is None:
            print("No raw data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Raw scene brightness if present
        if 'scene_brightness' in self.raw_data.columns:
            axes[0, 0].plot(self.raw_data['scene_brightness'].values)
            axes[0, 0].set_title('Raw Scene Brightness (proxy)')
            axes[0, 0].set_xlabel('Time Index')
            axes[0, 0].set_ylabel('Normalized Brightness (arbitrary)')
        else:
            first_col = self.raw_data.select_dtypes(include=[np.number]).columns[0]
            axes[0, 0].plot(self.raw_data[first_col].values)
            axes[0, 0].set_title(f'Raw Signal: {first_col}')
            axes[0, 0].set_xlabel('Time Index')
            axes[0, 0].set_ylabel('Value')
        
        # Normalized data
        if self.normalized_data is not None:
            axes[0, 1].plot(self.normalized_data)
            axes[0, 1].set_title('Normalized Scene Signal')
            axes[0, 1].set_xlabel('Time Index')
            axes[0, 1].set_ylabel('Normalized Value')
        
        # Sample sequences
        if self.sequences:
            # Plot first few sequences
            for i in range(min(5, len(self.sequences))):
                axes[1, 0].plot(self.sequences[i], alpha=0.7, label=f'Seq {i+1}')
            axes[1, 0].set_title('Sample Sequences')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Normalized Value')
            axes[1, 0].legend()
        
        # Data distribution
        if self.normalized_data is not None:
            axes[1, 1].hist(self.normalized_data, bins=50, alpha=0.7)
            axes[1, 1].set_title('Data Distribution')
            axes[1, 1].set_xlabel('Normalized Value')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def test_data_processor():
    """Test the data processor with synthetic data"""
    processor = DataProcessor()
    
    # Load synthetic data
    data = processor.load_data()
    
    # Normalize data
    normalized_data = processor.normalize_data(data)
    
    # Generate sequences
    sequences = processor.generate_sequences()
    
    # Split data
    train_seq, test_seq = processor.split_data()
    
    # Create data loaders
    train_loader, test_loader = processor.create_data_loaders()
    
    print("Data processing test completed successfully!")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    return processor


if __name__ == "__main__":
    test_data_processor()
