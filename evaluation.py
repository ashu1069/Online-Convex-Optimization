"""
Evaluation and visualization utilities for vision-based controller tuning
Adds flicker/stability metrics alongside OCO costs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import pandas as pd
from robd_algorithm import ROBDAgent
from ml_model import MLOCOPredictor
from hybrid_algorithm import ExpertCalibratedLearning, AdaptiveHybridAlgorithm


class OCOEvaluator:
    """
    Comprehensive evaluator for OCO algorithms
    """
    
    def __init__(self, m: float = 5.0):
        """
        Initialize evaluator
        
        Args:
            m: hitting cost parameter
        """
        self.m = m
        self.results = {}
        
    def evaluate_algorithm(self, algorithm, contexts: List[float], 
                          algorithm_name: str) -> Dict:
        """
        Evaluate a single algorithm
        
        Args:
            algorithm: algorithm instance
            contexts: test contexts
            algorithm_name: name for identification
            
        Returns:
            evaluation results
        """
        if hasattr(algorithm, 'run_sequence'):
            # For hybrid algorithms
            actions, results = algorithm.run_sequence(contexts)
            total_cost = results['total_cost']
        else:
            # For R-OBD and ML algorithms
            actions, total_cost = algorithm.predict(contexts)
            results = {'actions': actions, 'total_cost': total_cost}
        
        # Compute additional metrics
        avg_cost = total_cost / len(contexts)
        
        # Compute cost components
        hitting_costs = []
        switching_costs = []
        
        for t, (action, context) in enumerate(zip(actions, contexts)):
            hitting_cost = (self.m / 2) * (action - context) ** 2
            hitting_costs.append(hitting_cost)
            
            if t > 0:
                switching_cost = (1 / 2) * (action - actions[t-1]) ** 2
            else:
                switching_cost = 0.0
            switching_costs.append(switching_cost)
        
        # Flicker metric: mean absolute action delta (lower is better)
        deltas = [abs(actions[t] - actions[t-1]) for t in range(1, len(actions))]
        avg_delta = float(np.mean(deltas)) if deltas else 0.0
        peak_delta = float(np.max(deltas)) if deltas else 0.0

        evaluation_results = {
            'algorithm_name': algorithm_name,
            'actions': actions,
            'total_cost': total_cost,
            'avg_cost': avg_cost,
            'hitting_costs': hitting_costs,
            'switching_costs': switching_costs,
            'total_hitting_cost': sum(hitting_costs),
            'total_switching_cost': sum(switching_costs),
            'cost_ratio': sum(hitting_costs) / sum(switching_costs) if sum(switching_costs) > 0 else float('inf'),
            'avg_action_delta': avg_delta,
            'peak_action_delta': peak_delta
        }
        
        self.results[algorithm_name] = evaluation_results
        return evaluation_results
    
    def compare_algorithms(self, algorithms: Dict, contexts: List[float]) -> pd.DataFrame:
        """
        Compare multiple algorithms on the same contexts
        
        Args:
            algorithms: dict of {name: algorithm_instance}
            contexts: test contexts
            
        Returns:
            comparison dataframe
        """
        comparison_data = []
        
        for name, algorithm in algorithms.items():
            results = self.evaluate_algorithm(algorithm, contexts, name)
            comparison_data.append({
                'Algorithm': name,
                'Total Cost': results['total_cost'],
                'Average Cost': results['avg_cost'],
                'Hitting Cost': results['total_hitting_cost'],
                'Switching Cost': results['total_switching_cost'],
                'Cost Ratio': results['cost_ratio'],
                'Avg Action Δ': results['avg_action_delta'],
                'Peak Action Δ': results['peak_action_delta']
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_cost_distribution(self, save_path: str = None):
        """
        Plot cost distribution for all algorithms
        
        Args:
            save_path: path to save the plot
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total cost comparison
        algorithms = list(self.results.keys())
        total_costs = [self.results[alg]['total_cost'] for alg in algorithms]
        
        axes[0, 0].bar(algorithms, total_costs)
        axes[0, 0].set_title('Total Cost Comparison')
        axes[0, 0].set_ylabel('Total Cost')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Average cost comparison
        avg_costs = [self.results[alg]['avg_cost'] for alg in algorithms]
        axes[0, 1].bar(algorithms, avg_costs)
        axes[0, 1].set_title('Average Cost Comparison')
        axes[0, 1].set_ylabel('Average Cost')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cost breakdown
        hitting_costs = [self.results[alg]['total_hitting_cost'] for alg in algorithms]
        switching_costs = [self.results[alg]['total_switching_cost'] for alg in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, hitting_costs, width, label='Hitting Cost', alpha=0.8)
        axes[1, 0].bar(x + width/2, switching_costs, width, label='Switching Cost', alpha=0.8)
        axes[1, 0].set_title('Cost Breakdown')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(algorithms, rotation=45)
        axes[1, 0].legend()
        
        # Cost ratio
        cost_ratios = [self.results[alg]['cost_ratio'] for alg in algorithms]
        axes[1, 1].bar(algorithms, cost_ratios)
        axes[1, 1].set_title('Hitting/Switching Cost Ratio')
        axes[1, 1].set_ylabel('Cost Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_action_sequences(self, contexts: List[float], save_path: str = None):
        """
        Plot action sequences for all algorithms
        
        Args:
            contexts: test contexts
            save_path: path to save the plot
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (alg_name, results) in enumerate(self.results.items()):
            if i >= 4:  # Limit to 4 subplots
                break
                
            ax = axes[i]
            actions = results['actions']
            hitting_costs = results['hitting_costs']
            switching_costs = results['switching_costs']
            
            # Plot contexts and actions
            time_steps = range(len(contexts))
            ax.plot(time_steps, contexts, 'o-', label='Context', alpha=0.7)
            ax.plot(time_steps, actions, 's-', label='Actions', alpha=0.7)
            
            # Add cost information
            ax2 = ax.twinx()
            ax2.bar(time_steps, hitting_costs, alpha=0.3, label='Hitting Cost', color='red')
            ax2.bar(time_steps, switching_costs, alpha=0.3, label='Switching Cost', color='blue')
            
            ax.set_title(f'{alg_name}\nTotal Cost: {results["total_cost"]:.3f}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_hybrid_weights(self, hybrid_results: Dict, save_path: str = None):
        """
        Plot weight evolution for hybrid algorithms
        
        Args:
            hybrid_results: results from hybrid algorithm
            save_path: path to save the plot
        """
        if 'weight_history' not in hybrid_results:
            print("No weight history to plot")
            return
        
        weight_history = hybrid_results['weight_history']
        
        if isinstance(weight_history[0], np.ndarray):
            # Expert-calibrated learning weights
            weights_array = np.array(weight_history)
            plt.figure(figsize=(10, 6))
            plt.plot(weights_array[:, 0], label='ML Weight', linewidth=2)
            plt.plot(weights_array[:, 1], label='R-OBD Weight', linewidth=2)
            plt.title('Expert Weight Evolution')
            plt.xlabel('Time Step')
            plt.ylabel('Weight')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # Adaptive hybrid weights
            plt.figure(figsize=(10, 6))
            plt.plot(weight_history, label='ML Weight', linewidth=2)
            plt.title('Adaptive ML Weight Evolution')
            plt.xlabel('Time Step')
            plt.ylabel('ML Weight')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_flicker_analysis(self, save_path: str = None):
        """
        Plot flicker analysis comparing action stability across algorithms
        
        Args:
            save_path: path to save the plot
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Average action delta comparison
        algorithms = list(self.results.keys())
        avg_deltas = [self.results[alg]['avg_action_delta'] for alg in algorithms]
        peak_deltas = [self.results[alg]['peak_action_delta'] for alg in algorithms]
        
        # Bar plot for average deltas
        bars1 = axes[0, 0].bar(algorithms, avg_deltas, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Average Action Delta (Flicker Metric)')
        axes[0, 0].set_ylabel('Average |Δaction|')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, avg_deltas):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Bar plot for peak deltas
        bars2 = axes[0, 1].bar(algorithms, peak_deltas, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Peak Action Delta (Worst Flicker)')
        axes[0, 1].set_ylabel('Peak |Δaction|')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, peak_deltas):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Flicker vs Performance scatter
        total_costs = [self.results[alg]['total_cost'] for alg in algorithms]
        axes[1, 0].scatter(avg_deltas, total_costs, s=100, alpha=0.7)
        for i, alg in enumerate(algorithms):
            axes[1, 0].annotate(alg, (avg_deltas[i], total_costs[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('Average Action Delta')
        axes[1, 0].set_ylabel('Total Cost')
        axes[1, 0].set_title('Flicker vs Performance Trade-off')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Stability score (lower is better)
        stability_scores = [avg + peak for avg, peak in zip(avg_deltas, peak_deltas)]
        bars3 = axes[1, 1].bar(algorithms, stability_scores, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Overall Stability Score')
        axes[1, 1].set_ylabel('Avg + Peak Delta')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars3, stability_scores):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_summary(self, save_path: str = None):
        """
        Plot comprehensive performance summary
        
        Args:
            save_path: path to save the plot
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        algorithms = list(self.results.keys())
        
        # Total cost comparison
        total_costs = [self.results[alg]['total_cost'] for alg in algorithms]
        bars1 = axes[0, 0].bar(algorithms, total_costs, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Total Cost Comparison')
        axes[0, 0].set_ylabel('Total Cost')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, total_costs):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Cost breakdown
        hitting_costs = [self.results[alg]['total_hitting_cost'] for alg in algorithms]
        switching_costs = [self.results[alg]['total_switching_cost'] for alg in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, hitting_costs, width, label='Hitting Cost', alpha=0.8)
        axes[0, 1].bar(x + width/2, switching_costs, width, label='Switching Cost', alpha=0.8)
        axes[0, 1].set_title('Cost Breakdown')
        axes[0, 1].set_ylabel('Cost')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(algorithms, rotation=45)
        axes[0, 1].legend()
        
        # Flicker metrics
        avg_deltas = [self.results[alg]['avg_action_delta'] for alg in algorithms]
        peak_deltas = [self.results[alg]['peak_action_delta'] for alg in algorithms]
        
        axes[0, 2].bar(x - width/2, avg_deltas, width, label='Avg Delta', alpha=0.8)
        axes[0, 2].bar(x + width/2, peak_deltas, width, label='Peak Delta', alpha=0.8)
        axes[0, 2].set_title('Flicker Analysis')
        axes[0, 2].set_ylabel('Action Delta')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(algorithms, rotation=45)
        axes[0, 2].legend()
        
        # Performance radar chart
        metrics = ['Total Cost', 'Hitting Cost', 'Switching Cost', 'Avg Delta', 'Peak Delta']
        normalized_data = []
        
        for alg in algorithms:
            values = [
                self.results[alg]['total_cost'],
                self.results[alg]['total_hitting_cost'],
                self.results[alg]['total_switching_cost'],
                self.results[alg]['avg_action_delta'],
                self.results[alg]['peak_action_delta']
            ]
            # Normalize (lower is better for all metrics)
            max_vals = [max(self.results[a]['total_cost'] for a in algorithms),
                       max(self.results[a]['total_hitting_cost'] for a in algorithms),
                       max(self.results[a]['total_switching_cost'] for a in algorithms),
                       max(self.results[a]['avg_action_delta'] for a in algorithms),
                       max(self.results[a]['peak_action_delta'] for a in algorithms)]
            
            normalized = [1 - (val / max_val) for val, max_val in zip(values, max_vals)]
            normalized_data.append(normalized)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, (alg, data) in enumerate(zip(algorithms, normalized_data)):
            data += data[:1]  # Complete the circle
            axes[1, 0].plot(angles, data, 'o-', linewidth=2, label=alg)
            axes[1, 0].fill(angles, data, alpha=0.25)
        
        axes[1, 0].set_xticks(angles[:-1])
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].set_title('Performance Radar Chart')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Efficiency score (cost per stability)
        efficiency_scores = [total_cost / (avg_delta + 0.001) for total_cost, avg_delta in zip(total_costs, avg_deltas)]
        bars4 = axes[1, 1].bar(algorithms, efficiency_scores, color='gold', alpha=0.7)
        axes[1, 1].set_title('Efficiency Score (Cost/Stability)')
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars4, efficiency_scores):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Algorithm ranking
        rankings = []
        for alg in algorithms:
            rank = 1
            for other_alg in algorithms:
                if self.results[other_alg]['total_cost'] < self.results[alg]['total_cost']:
                    rank += 1
            rankings.append(rank)
        
        colors = ['gold', 'silver', '#CD7F32', 'lightgray'][:len(algorithms)]
        bars5 = axes[1, 2].bar(algorithms, rankings, color=colors, alpha=0.7)
        axes[1, 2].set_title('Algorithm Ranking')
        axes[1, 2].set_ylabel('Rank (1=Best)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].set_ylim(0, len(algorithms) + 1)
        
        # Add rank labels
        for bar, rank in zip(bars5, rankings):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'#{rank}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_algorithm_heatmap(self, save_path: str = None):
        """
        Plot algorithm comparison heatmap
        
        Args:
            save_path: path to save the plot
        """
        if not self.results:
            print("No results to plot")
            return
        
        # Prepare data for heatmap
        algorithms = list(self.results.keys())
        metrics = ['Total Cost', 'Avg Cost', 'Hitting Cost', 'Switching Cost', 
                  'Cost Ratio', 'Avg Delta', 'Peak Delta']
        
        data_matrix = []
        for alg in algorithms:
            row = [
                self.results[alg]['total_cost'],
                self.results[alg]['avg_cost'],
                self.results[alg]['total_hitting_cost'],
                self.results[alg]['total_switching_cost'],
                self.results[alg]['cost_ratio'],
                self.results[alg]['avg_action_delta'],
                self.results[alg]['peak_action_delta']
            ]
            data_matrix.append(row)
        
        # Normalize data for better visualization
        data_array = np.array(data_matrix)
        normalized_data = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0) + 1e-12)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw data heatmap
        im1 = ax1.imshow(data_array, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.set_yticks(range(len(algorithms)))
        ax1.set_yticklabels(algorithms)
        ax1.set_title('Raw Performance Metrics')
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(metrics)):
                text = ax1.text(j, i, f'{data_array[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Normalized heatmap
        im2 = ax2.imshow(normalized_data, cmap='RdYlGn_r', aspect='auto')
        ax2.set_xticks(range(len(metrics)))
        ax2.set_xticklabels(metrics, rotation=45, ha='right')
        ax2.set_yticks(range(len(algorithms)))
        ax2.set_yticklabels(algorithms)
        ax2.set_title('Normalized Performance Metrics')
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(metrics)):
                text = ax2.text(j, i, f'{normalized_data[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbars
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive evaluation report
        
        Returns:
            formatted report string
        """
        if not self.results:
            return "No results to report"
        
        report = "=" * 60 + "\n"
        report += "OCO ALGORITHM EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Summary table
        report += "PERFORMANCE SUMMARY:\n"
        report += "-" * 40 + "\n"
        
        for alg_name, results in self.results.items():
            report += f"\n{alg_name}:\n"
            report += f"  Total Cost: {results['total_cost']:.6f}\n"
            report += f"  Average Cost: {results['avg_cost']:.6f}\n"
            report += f"  Hitting Cost: {results['total_hitting_cost']:.6f}\n"
            report += f"  Switching Cost: {results['total_switching_cost']:.6f}\n"
            report += f"  Cost Ratio: {results['cost_ratio']:.3f}\n"
        
        # Best algorithm
        best_alg = min(self.results.keys(), key=lambda x: self.results[x]['total_cost'])
        report += f"\nBEST ALGORITHM: {best_alg}\n"
        report += f"Best Total Cost: {self.results[best_alg]['total_cost']:.6f}\n"
        
        # Performance comparison
        report += "\nPERFORMANCE COMPARISON:\n"
        report += "-" * 40 + "\n"
        
        costs = [results['total_cost'] for results in self.results.values()]
        min_cost = min(costs)
        
        for alg_name, results in self.results.items():
            improvement = ((results['total_cost'] - min_cost) / min_cost) * 100
            report += f"{alg_name}: {improvement:+.2f}% vs best\n"
        
        return report


def run_comprehensive_evaluation():
    """
    Run comprehensive evaluation of all algorithms
    """
    # Create test contexts
    np.random.seed(42)
    contexts = np.random.uniform(0, 1, 50).tolist()
    
    # Initialize algorithms
    robd_agent = ROBDAgent()
    ml_predictor = MLOCOPredictor()
    hybrid_alg = ExpertCalibratedLearning(ml_predictor, robd_agent)
    adaptive_hybrid = AdaptiveHybridAlgorithm(ml_predictor, robd_agent)
    
    # Create evaluator
    evaluator = OCOEvaluator()
    
    # Evaluate all algorithms
    algorithms = {
        'R-OBD': robd_agent,
        'ML Predictor': ml_predictor,
        'Expert-Calibrated': hybrid_alg,
        'Adaptive Hybrid': adaptive_hybrid
    }
    
    # Run evaluation
    comparison_df = evaluator.compare_algorithms(algorithms, contexts)
    
    # Generate plots
    evaluator.plot_cost_distribution('cost_comparison.png')
    evaluator.plot_action_sequences(contexts, 'action_sequences.png')
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
    
    # Save results
    comparison_df.to_csv('algorithm_comparison.csv', index=False)
    
    with open('evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print("Evaluation completed! Results saved to:")
    print("- cost_comparison.png")
    print("- action_sequences.png") 
    print("- algorithm_comparison.csv")
    print("- evaluation_report.txt")
    
    return evaluator, comparison_df


if __name__ == "__main__":
    run_comprehensive_evaluation()
