"""
Generate synthetic vision signals for controller tuning
Creates realistic scene brightness and feature density patterns
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_vision_signals(
    duration_hours: int = 19 * 24,  # 19 days
    sample_rate_hz: float = 1.0,    # 1 sample per hour for simplicity
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic vision signals for controller tuning
    
    Args:
        duration_hours: total duration in hours
        sample_rate_hz: samples per hour
        seed: random seed for reproducibility
        
    Returns:
        DataFrame with time_index, scene_brightness, feature_density
    """
    np.random.seed(seed)
    
    # Time indices
    time_indices = np.arange(0, duration_hours)
    
    # Scene brightness: diurnal cycle + weekly variation + transients
    base_brightness = 0.6
    daily_cycle = 0.3 * np.clip(np.sin(2 * np.pi * time_indices / 24) + 0.2, 0, 1)
    weekly_cycle = 0.1 * (np.sin(2 * np.pi * time_indices / (24 * 7)) + 1) / 2
    transients = 0.15 * np.random.rand(len(time_indices))  # sudden lighting changes
    
    scene_brightness = base_brightness * (0.5 + daily_cycle) + weekly_cycle + transients
    scene_brightness = np.clip(scene_brightness, 0.01, 1.0)
    
    # Feature density: anticorrelated with brightness + noise
    feature_density = 0.7 - 0.4 * daily_cycle + 0.15 * np.random.randn(len(time_indices))
    feature_density = np.clip(feature_density, 0.01, 1.0)
    
    df = pd.DataFrame({
        'time_index': time_indices,
        'scene_brightness': scene_brightness,
        'feature_density': feature_density
    })
    
    return df


def create_exposure_targets(scene_brightness: np.ndarray) -> np.ndarray:
    """
    Convert scene brightness to exposure gain targets
    
    Args:
        scene_brightness: normalized brightness values [0, 1]
        
    Returns:
        exposure gain targets (higher for darker scenes)
    """
    # Inverse relationship: darker scenes need higher gain
    exposure_gain = 1.0 / (scene_brightness + 0.1)  # Add small epsilon
    exposure_gain = np.clip(exposure_gain, 0.5, 5.0)  # Reasonable range
    return exposure_gain


def create_detection_thresholds(feature_density: np.ndarray) -> np.ndarray:
    """
    Convert feature density to detection threshold targets
    
    Args:
        feature_density: normalized feature density [0, 1]
        
    Returns:
        detection threshold targets (lower for more features)
    """
    # Inverse relationship: more features allow lower thresholds
    detection_threshold = 1.0 - feature_density + 0.1  # Add small epsilon
    detection_threshold = np.clip(detection_threshold, 0.1, 0.9)  # Reasonable range
    return detection_threshold


def generate_controller_dataset(
    duration_hours: int = 19 * 24,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate complete controller tuning dataset
    
    Args:
        duration_hours: total duration in hours
        seed: random seed for reproducibility
        
    Returns:
        DataFrame with scene signals and controller targets
    """
    # Generate base vision signals
    vision_df = generate_vision_signals(duration_hours, seed=seed)
    
    # Create controller targets
    exposure_targets = create_exposure_targets(vision_df['scene_brightness'].values)
    detection_targets = create_detection_thresholds(vision_df['feature_density'].values)
    
    # Add controller targets to dataframe
    vision_df['exposure_gain_target'] = exposure_targets
    vision_df['detection_threshold_target'] = detection_targets
    
    return vision_df


def main():
    """Generate and save vision signals dataset"""
    print("Generating vision signals dataset...")
    
    # Generate dataset
    df = generate_controller_dataset(duration_hours=19*24, seed=42)
    
    # Save to CSV
    output_file = 'vision_signals.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Generated {len(df)} samples")
    print(f"Scene brightness range: [{df['scene_brightness'].min():.3f}, {df['scene_brightness'].max():.3f}]")
    print(f"Feature density range: [{df['feature_density'].min():.3f}, {df['feature_density'].max():.3f}]")
    print(f"Exposure gain range: [{df['exposure_gain_target'].min():.3f}, {df['exposure_gain_target'].max():.3f}]")
    print(f"Detection threshold range: [{df['detection_threshold_target'].min():.3f}, {df['detection_threshold_target'].max():.3f}]")
    print(f"Saved to {output_file}")
    
    return df


if __name__ == "__main__":
    main()
