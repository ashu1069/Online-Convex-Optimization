"""
Regularized Online Balanced Descent (R-OBD) Algorithm Implementation
Based on: "Beyond online balanced descent: An optimal algorithm for smoothed online optimization"
"""

import numpy as np
import torch
from typing import List, Tuple


class ROBDAgent:
    """
    Regularized Online Balanced Descent (R-OBD) Algorithm
    
    The algorithm maintains two regularizers:
    - λ1: for hitting cost regularization
    - λ2: for switching cost regularization
    """
    
    def __init__(self, m: float = 5.0, lambda1: float = None, lambda2: float = None):
        """
        Initialize R-OBD agent
        
        Args:
            m: hitting cost parameter (default 5.0 as specified)
            lambda1: regularization parameter for hitting cost
            lambda2: regularization parameter for switching cost
        """
        self.m = m
        
        # Set optimal hyperparameters from Theorem 4
        # For the case where both costs are squared l2-norm
        if lambda1 is None:
            self.lambda1 = 1.0  # Optimal value for hitting cost regularization
        else:
            self.lambda1 = lambda1
            
        if lambda2 is None:
            self.lambda2 = 1.0  # Optimal value for switching cost regularization  
        else:
            self.lambda2 = lambda2
            
        self.action_history = []
        self.context_history = []
        
    def predict_action(self, context: float, prev_action: float = None) -> float:
        """
        Predict next action using R-OBD algorithm
        
        Args:
            context: current context y_t
            prev_action: previous action x_{t-1} (None for first step)
            
        Returns:
            predicted action x_t
        """
        if prev_action is None:
            # First action: minimize hitting cost only
            action = context
        else:
            # R-OBD update rule
            # The algorithm balances hitting cost and switching cost
            # Using the regularized update from the paper
            
            # Compute the gradient of the regularized objective
            # ∇f(x) = m * (x - y) + λ1 * (x - y) + λ2 * (x - x_prev)
            # Setting gradient to zero: (m + λ1) * x - m * y - λ1 * y - λ2 * x_prev = 0
            # Solving: x = (m * y + λ1 * y + λ2 * x_prev) / (m + λ1 + λ2)
            
            numerator = (self.m + self.lambda1) * context + self.lambda2 * prev_action
            denominator = self.m + self.lambda1 + self.lambda2
            
            action = numerator / denominator
            
        return action
    
    def update(self, context: float, action: float):
        """Update agent with new context and action"""
        self.context_history.append(context)
        self.action_history.append(action)
    
    def compute_cost(self, action: float, context: float, prev_action: float = None) -> float:
        """
        Compute the total cost for a given action
        
        Args:
            action: current action x_t
            context: current context y_t  
            prev_action: previous action x_{t-1}
            
        Returns:
            total cost (hitting + switching)
        """
        hitting_cost = (self.m / 2) * (action - context) ** 2
        
        if prev_action is None:
            switching_cost = 0.0
        else:
            switching_cost = (1 / 2) * (action - prev_action) ** 2
            
        return hitting_cost + switching_cost
    
    def get_total_cost(self, contexts: List[float]) -> Tuple[List[float], float]:
        """
        Run R-OBD on a sequence of contexts and return actions and total cost
        
        Args:
            contexts: list of contexts y_1, y_2, ..., y_T
            
        Returns:
            tuple of (actions, total_cost)
        """
        actions = []
        total_cost = 0.0
        
        for t, context in enumerate(contexts):
            prev_action = actions[-1] if actions else None
            action = self.predict_action(context, prev_action)
            actions.append(action)
            
            cost = self.compute_cost(action, context, prev_action)
            total_cost += cost
            
            self.update(context, action)
            
        return actions, total_cost

    def predict(self, contexts: List[float]) -> Tuple[List[float], float]:
        """
        Wrapper to conform to evaluator interface.
        Returns actions and total cost for the given contexts.
        """
        return self.get_total_cost(contexts)


def test_robd():
    """Test the R-OBD implementation with a simple example"""
    # Create a simple test case
    contexts = [0.2, 0.5, 0.8, 0.3, 0.7]
    
    agent = ROBDAgent()
    actions, total_cost = agent.get_total_cost(contexts)
    
    print("R-OBD Test Results:")
    print(f"Contexts: {contexts}")
    print(f"Actions: {[f'{a:.3f}' for a in actions]}")
    print(f"Total Cost: {total_cost:.3f}")
    
    return actions, total_cost


if __name__ == "__main__":
    test_robd()
