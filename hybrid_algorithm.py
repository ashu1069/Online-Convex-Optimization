"""
Hybrid Algorithm: Expert-Calibrated Learning for Online Optimization
Combines ML predictions with R-OBD using the technique from paper [5]
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from robd_algorithm import ROBDAgent
from ml_model import MLOCOPredictor


class ExpertCalibratedLearning:
    """
    Expert-Calibrated Learning Algorithm
    Combines ML predictions with R-OBD expert using calibrated weights
    """
    
    def __init__(self, ml_predictor: MLOCOPredictor, robd_agent: ROBDAgent, 
                 learning_rate: float = 0.01, beta: float = 1.0):
        """
        Initialize hybrid algorithm
        
        Args:
            ml_predictor: trained ML model
            robd_agent: R-OBD expert
            learning_rate: learning rate for weight updates
            beta: regularization parameter for weight updates
        """
        self.ml_predictor = ml_predictor
        self.robd_agent = robd_agent
        self.learning_rate = learning_rate
        self.beta = beta
        
        # Initialize weights for expert combination
        self.weights = np.array([0.5, 0.5])  # Equal weights initially
        self.weight_history = []
        self.cost_history = []
        
    def predict_action(self, context: float, prev_action: float = None, 
                      prev_contexts: List[float] = None) -> Tuple[float, dict]:
        """
        Predict action using hybrid approach
        
        Args:
            context: current context y_t
            prev_action: previous action x_{t-1}
            prev_contexts: previous contexts for ML model
            
        Returns:
            tuple of (predicted_action, prediction_info)
        """
        # Get ML prediction
        if prev_contexts is not None:
            ml_actions, _ = self.ml_predictor.predict(prev_contexts + [context])
            ml_prediction = ml_actions[-1]
        else:
            ml_prediction = context  # Fallback to context
        
        # Get R-OBD prediction
        robd_prediction = self.robd_agent.predict_action(context, prev_action)
        
        # Combine predictions using current weights
        combined_action = self.weights[0] * ml_prediction + self.weights[1] * robd_prediction
        
        # Store prediction info
        prediction_info = {
            'ml_prediction': ml_prediction,
            'robd_prediction': robd_prediction,
            'combined_action': combined_action,
            'weights': self.weights.copy()
        }
        
        return combined_action, prediction_info
    
    def update_weights(self, action: float, context: float, prev_action: float = None):
        """
        Update expert weights based on performance
        
        Args:
            action: actual action taken
            context: current context
            prev_action: previous action
        """
        # Compute costs for each expert
        ml_cost = self._compute_expert_cost(action, context, prev_action, 'ml')
        robd_cost = self._compute_expert_cost(action, context, prev_action, 'robd')
        
        # Update weights using exponential weights algorithm
        # w_i = w_i * exp(-η * cost_i) / Z
        # where Z is normalization factor
        
        ml_loss = ml_cost
        robd_loss = robd_cost
        
        # Update weights
        self.weights[0] *= np.exp(-self.learning_rate * ml_loss)
        self.weights[1] *= np.exp(-self.learning_rate * robd_loss)
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
        
        # Store history
        self.weight_history.append(self.weights.copy())
        self.cost_history.append({'ml': ml_cost, 'robd': robd_cost})
    
    def _compute_expert_cost(self, action: float, context: float, prev_action: float, 
                           expert_type: str) -> float:
        """
        Compute cost for a specific expert
        
        Args:
            action: actual action taken
            context: current context
            prev_action: previous action
            expert_type: 'ml' or 'robd'
            
        Returns:
            expert cost
        """
        m = self.robd_agent.m
        
        # Hitting cost
        hitting_cost = (m / 2) * (action - context) ** 2
        
        # Switching cost
        if prev_action is not None:
            switching_cost = (1 / 2) * (action - prev_action) ** 2
        else:
            switching_cost = 0.0
        
        return hitting_cost + switching_cost
    
    def run_sequence(self, contexts: List[float]) -> Tuple[List[float], dict]:
        """
        Run hybrid algorithm on a sequence of contexts
        
        Args:
            contexts: list of contexts y_1, y_2, ..., y_T
            
        Returns:
            tuple of (actions, results_dict)
        """
        actions = []
        prediction_info_list = []
        total_cost = 0.0
        
        for t, context in enumerate(contexts):
            prev_action = actions[-1] if actions else None
            prev_contexts = contexts[:t] if t > 0 else []
            
            # Predict action
            action, prediction_info = self.predict_action(context, prev_action, prev_contexts)
            actions.append(action)
            prediction_info_list.append(prediction_info)
            
            # Compute cost
            cost = self._compute_expert_cost(action, context, prev_action, 'combined')
            total_cost += cost
            
            # Update weights (except for first step)
            if t > 0:
                self.update_weights(action, context, prev_action)
        
        results = {
            'actions': actions,
            'total_cost': total_cost,
            'prediction_info': prediction_info_list,
            'final_weights': self.weights.copy(),
            'weight_history': self.weight_history.copy(),
            'cost_history': self.cost_history.copy()
        }
        
        return actions, results


class AdaptiveHybridAlgorithm:
    """
    Adaptive version of hybrid algorithm with dynamic weight adjustment
    """
    
    def __init__(self, ml_predictor: MLOCOPredictor, robd_agent: ROBDAgent,
                 initial_ml_weight: float = 0.5, adaptation_rate: float = 0.1):
        """
        Initialize adaptive hybrid algorithm
        
        Args:
            ml_predictor: trained ML model
            robd_agent: R-OBD expert
            initial_ml_weight: initial weight for ML predictor
            adaptation_rate: rate of weight adaptation
        """
        self.ml_predictor = ml_predictor
        self.robd_agent = robd_agent
        self.ml_weight = initial_ml_weight
        self.adaptation_rate = adaptation_rate
        
        self.performance_history = []
        self.weight_history = []
        
    def predict_action(self, context: float, prev_action: float = None,
                      prev_contexts: List[float] = None) -> Tuple[float, dict]:
        """
        Predict action using adaptive hybrid approach
        
        Args:
            context: current context y_t
            prev_action: previous action x_{t-1}
            prev_contexts: previous contexts for ML model
            
        Returns:
            tuple of (predicted_action, prediction_info)
        """
        # Get ML prediction
        if prev_contexts is not None and len(prev_contexts) > 0:
            ml_actions, _ = self.ml_predictor.predict(prev_contexts + [context])
            ml_prediction = ml_actions[-1]
        else:
            ml_prediction = context
        
        # Get R-OBD prediction
        robd_prediction = self.robd_agent.predict_action(context, prev_action)
        
        # Combine predictions
        combined_action = self.ml_weight * ml_prediction + (1 - self.ml_weight) * robd_prediction
        
        prediction_info = {
            'ml_prediction': ml_prediction,
            'robd_prediction': robd_prediction,
            'combined_action': combined_action,
            'ml_weight': self.ml_weight
        }
        
        return combined_action, prediction_info
    
    def adapt_weights(self, recent_performance: dict):
        """
        Adapt weights based on recent performance
        
        Args:
            recent_performance: dict with 'ml_performance' and 'robd_performance'
        """
        ml_perf = recent_performance.get('ml_performance', 0.5)
        robd_perf = recent_performance.get('robd_performance', 0.5)
        
        # Adjust ML weight based on relative performance
        if ml_perf < robd_perf:
            # ML is performing better, increase its weight
            self.ml_weight = min(0.9, self.ml_weight + self.adaptation_rate)
        else:
            # R-OBD is performing better, decrease ML weight
            self.ml_weight = max(0.1, self.ml_weight - self.adaptation_rate)
        
        self.weight_history.append(self.ml_weight)
    
    def run_sequence(self, contexts: List[float], adaptation_window: int = 10) -> Tuple[List[float], dict]:
        """
        Run adaptive hybrid algorithm on a sequence
        
        Args:
            contexts: list of contexts
            adaptation_window: window size for performance evaluation
            
        Returns:
            tuple of (actions, results_dict)
        """
        actions = []
        prediction_info_list = []
        total_cost = 0.0
        ml_costs = []
        robd_costs = []
        
        for t, context in enumerate(contexts):
            prev_action = actions[-1] if actions else None
            prev_contexts = contexts[:t] if t > 0 else []
            
            # Predict action
            action, prediction_info = self.predict_action(context, prev_action, prev_contexts)
            actions.append(action)
            prediction_info_list.append(prediction_info)
            
            # Compute costs for both experts
            m = self.robd_agent.m
            hitting_cost = (m / 2) * (action - context) ** 2
            switching_cost = (1 / 2) * (action - prev_action) ** 2 if prev_action is not None else 0.0
            total_cost += hitting_cost + switching_cost
            
            # Store individual expert costs
            ml_cost = self._compute_ml_cost(prediction_info['ml_prediction'], context, prev_action)
            robd_cost = self._compute_robd_cost(prediction_info['robd_prediction'], context, prev_action)
            
            ml_costs.append(ml_cost)
            robd_costs.append(robd_cost)
            
            # Adapt weights periodically
            if t > 0 and t % adaptation_window == 0:
                recent_ml_perf = np.mean(ml_costs[-adaptation_window:])
                recent_robd_perf = np.mean(robd_costs[-adaptation_window:])
                
                self.adapt_weights({
                    'ml_performance': recent_ml_perf,
                    'robd_performance': recent_robd_perf
                })
        
        results = {
            'actions': actions,
            'total_cost': total_cost,
            'prediction_info': prediction_info_list,
            'final_ml_weight': self.ml_weight,
            'weight_history': self.weight_history.copy(),
            'ml_costs': ml_costs,
            'robd_costs': robd_costs
        }
        
        return actions, results
    
    def _compute_ml_cost(self, ml_action: float, context: float, prev_action: float) -> float:
        """Compute cost for ML prediction"""
        m = self.robd_agent.m
        hitting_cost = (m / 2) * (ml_action - context) ** 2
        switching_cost = (1 / 2) * (ml_action - prev_action) ** 2 if prev_action is not None else 0.0
        return hitting_cost + switching_cost
    
    def _compute_robd_cost(self, robd_action: float, context: float, prev_action: float) -> float:
        """Compute cost for R-OBD prediction"""
        m = self.robd_agent.m
        hitting_cost = (m / 2) * (robd_action - context) ** 2
        switching_cost = (1 / 2) * (robd_action - prev_action) ** 2 if prev_action is not None else 0.0
        return hitting_cost + switching_cost


def test_hybrid_algorithm():
    """Test the hybrid algorithm with a simple example"""
    # Create test contexts
    contexts = [0.2, 0.5, 0.8, 0.3, 0.7, 0.4, 0.9, 0.1]
    
    # Initialize components
    robd_agent = ROBDAgent()
    ml_predictor = MLOCOPredictor()
    
    # Test expert-calibrated learning
    hybrid_alg = ExpertCalibratedLearning(ml_predictor, robd_agent)
    actions, results = hybrid_alg.run_sequence(contexts)
    
    print("Hybrid Algorithm Test Results:")
    print(f"Contexts: {contexts}")
    print(f"Actions: {[f'{a:.3f}' for a in actions]}")
    print(f"Total Cost: {results['total_cost']:.3f}")
    print(f"Final Weights: ML={results['final_weights'][0]:.3f}, R-OBD={results['final_weights'][1]:.3f}")
    
    return actions, results


if __name__ == "__main__":
    test_hybrid_algorithm()
