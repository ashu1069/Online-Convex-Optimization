"""
Main script for vision-based controller tuning via SOCO + ML
Use case: online tuning of exposure/thresholds with flicker-aware smoothness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_processor import DataProcessor
from robd_algorithm import ROBDAgent
from ml_model import MLOCOPredictor
from hybrid_algorithm import ExpertCalibratedLearning, AdaptiveHybridAlgorithm
from evaluation import OCOEvaluator, run_comprehensive_evaluation


def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("VISION-BASED CONTROLLER TUNING (SOCO + ML)")
    print("=" * 80)
    print()
    
    # Step 1: Data Processing
    print("STEP 1: Data Processing")
    print("-" * 40)
    
    processor = DataProcessor()
    
    # Try to load vision data, fallback to synthetic scene signals
    try:
        data = processor.load_data('vision_signals.csv')
    except:
        print("vision_signals.csv not found, using synthetic data...")
        data = processor.load_data()
    
    # Normalize data to [0, 1]
    normalized_data = processor.normalize_data(data)
    
    # Generate sequences using 24-hour sliding window with step size 2
    sequences = processor.generate_sequences(window_size=24, step_size=2)
    
    # Split into train/test
    train_sequences, test_sequences = processor.split_data(train_ratio=0.8)
    
    print(f"Generated {len(sequences)} sequences")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Testing sequences: {len(test_sequences)}")
    print()
    
    # Step 2: Implement R-OBD Algorithm
    print("STEP 2: R-OBD Algorithm (Flicker-Aware)")
    print("-" * 40)
    
    robd_agent = ROBDAgent(m=5.0)
    print("R-OBD agent initialized with optimal hyperparameters")
    print(f"λ1 (hitting cost regularization): {robd_agent.lambda1}")
    print(f"λ2 (switching cost regularization): {robd_agent.lambda2}")
    print()
    
    # Step 3: Train ML Model
    print("STEP 3: ML Model Training (Context→Action)")
    print("-" * 40)
    
    # Create data loaders
    train_loader, test_loader = processor.create_data_loaders(batch_size=32)
    
    # Initialize ML predictor
    ml_predictor = MLOCOPredictor(
        input_dim=2,  # [prev_action, context]
        hidden_dim=64,
        num_layers=2,
        m=5.0,
        learning_rate=0.001
    )
    
    print("Training ML model...")
    train_losses = ml_predictor.train(train_loader, num_epochs=100, verbose=True)
    
    print("ML model training completed!")
    print()
    
    # Step 4: Hybrid Algorithm Implementation
    print("STEP 4: Hybrid Algorithm (Expert-Calibrated)")
    print("-" * 40)
    
    # Expert-calibrated learning
    expert_calibrated = ExpertCalibratedLearning(ml_predictor, robd_agent)
    
    # Adaptive hybrid algorithm
    adaptive_hybrid = AdaptiveHybridAlgorithm(ml_predictor, robd_agent)
    
    print("Hybrid algorithms initialized!")
    print()
    
    # Step 5: Evaluation and Comparison
    print("STEP 5: Performance Evaluation (Cost + Flicker)")
    print("-" * 40)
    
    # Create evaluator
    evaluator = OCOEvaluator(m=5.0)
    
    # Test on a sample sequence
    test_contexts = test_sequences[0].tolist()  # Use first test sequence
    
    print(f"Testing on sequence of length {len(test_contexts)}")
    
    # Evaluate all algorithms
    algorithms = {
        'R-OBD': robd_agent,
        'ML Predictor': ml_predictor,
        'Expert-Calibrated': expert_calibrated,
        'Adaptive Hybrid': adaptive_hybrid
    }
    
    # Run evaluation
    comparison_df = evaluator.compare_algorithms(algorithms, test_contexts)
    
    print("\nPERFORMANCE COMPARISON:")
    print(comparison_df.to_string(index=False))
    print()
    
    # Step 6: Enhanced Visualization Suite
    print("STEP 6: Generating Comprehensive Visualizations")
    print("-" * 40)
    
    # Plot cost distribution
    evaluator.plot_cost_distribution('results/cost_comparison.png')
    print("✓ Cost comparison plot saved")
    
    # Plot action sequences
    evaluator.plot_action_sequences(test_contexts, 'results/action_sequences.png')
    print("✓ Action sequences plot saved")
    
    # Plot hybrid weight evolution
    hybrid_results = evaluator.results.get('Expert-Calibrated', {})
    if 'prediction_info' in hybrid_results:
        evaluator.plot_hybrid_weights(hybrid_results, 'results/weight_evolution.png')
        print("✓ Weight evolution plot saved")
    
    # Generate flicker analysis plot
    evaluator.plot_flicker_analysis('results/flicker_analysis.png')
    print("✓ Flicker analysis plot saved")
    
    # Generate performance summary plot
    evaluator.plot_performance_summary('results/performance_summary.png')
    print("✓ Performance summary plot saved")
    
    # Generate algorithm comparison heatmap
    evaluator.plot_algorithm_heatmap('results/algorithm_heatmap.png')
    print("✓ Algorithm comparison heatmap saved")
    
    # Step 7: Generate Report
    print("STEP 7: Generating Final Report")
    print("-" * 40)
    
    report = evaluator.generate_report()
    print(report)
    
    # Save results
    comparison_df.to_csv('results/algorithm_comparison.csv', index=False)
    
    with open('results/evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print("✓ Results saved to results/ directory")
    print()
    
    # Step 8: Summary
    print("PROJECT SUMMARY")
    print("=" * 40)
    print("✓ R-OBD algorithm implemented with optimal hyperparameters")
    print("✓ ML model (LSTM) trained on historical data")
    print("✓ Hybrid algorithms combining ML and R-OBD")
    print("✓ Comprehensive evaluation and visualization")
    print("✓ Performance comparison completed")
    print()
    
    best_algorithm = comparison_df.loc[comparison_df['Total Cost'].idxmin(), 'Algorithm']
    best_cost = comparison_df.loc[comparison_df['Total Cost'].idxmin(), 'Total Cost']
    
    print(f"BEST PERFORMING ALGORITHM: {best_algorithm}")
    print(f"BEST TOTAL COST: {best_cost:.6f}")
    print()
    
    return evaluator, comparison_df


def run_quick_test():
    """
    Run a quick test with synthetic data
    """
    print("Running quick test with synthetic data...")
    
    # Create simple test data
    np.random.seed(42)
    contexts = np.random.uniform(0, 1, 20).tolist()
    
    # Initialize algorithms
    robd_agent = ROBDAgent()
    ml_predictor = MLOCOPredictor()
    hybrid_alg = ExpertCalibratedLearning(ml_predictor, robd_agent)
    
    # Test algorithms
    print("Testing R-OBD...")
    robd_actions, robd_cost = robd_agent.get_total_cost(contexts)
    print(f"R-OBD Total Cost: {robd_cost:.6f}")
    
    print("Testing ML Predictor...")
    ml_actions, ml_cost = ml_predictor.predict(contexts)
    print(f"ML Total Cost: {ml_cost:.6f}")
    
    print("Testing Hybrid Algorithm...")
    hybrid_actions, hybrid_results = hybrid_alg.run_sequence(contexts)
    print(f"Hybrid Total Cost: {hybrid_results['total_cost']:.6f}")
    
    # Compare results
    print("\nCOMPARISON:")
    print(f"R-OBD:     {robd_cost:.6f}")
    print(f"ML:        {ml_cost:.6f}")
    print(f"Hybrid:    {hybrid_results['total_cost']:.6f}")
    
    best = min(robd_cost, ml_cost, hybrid_results['total_cost'])
    if best == robd_cost:
        print("Best: R-OBD")
    elif best == ml_cost:
        print("Best: ML")
    else:
        print("Best: Hybrid")
    
    return robd_cost, ml_cost, hybrid_results['total_cost']


if __name__ == "__main__":
    import os
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run main pipeline
    try:
        evaluator, comparison_df = main()
        print("\n🎉 Project completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("Running quick test instead...")
        run_quick_test()
