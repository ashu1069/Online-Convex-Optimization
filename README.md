# Vision-Based Controller Tuning via SOCO + ML

This project targets a vision use case: online tuning of camera/controller parameters (e.g., exposure gain, detection threshold) from scene signals while minimizing flicker. We implement **Smoothed Online Convex Optimization (SOCO)** with ML augmentation: R-OBD for worst-case smoothness and an LSTM predictor for average-case performance, plus a hybrid approach.

## Problem Definition

In online convex optimization, we aim to minimize the following objective:

```
minimize Σ[t=1 to T] [m/2 * ||x_t - y_t||² + 1/2 * ||x_t - x_{t-1}||²]
```

Where:
- `x_t` is the action at time t (controller output; e.g., exposure gain)
- `y_t` is the context (scene signal; e.g., brightness proxy)
- `m` is the hitting cost parameter (set to 5.0)
- The first term is the **hitting cost** (tracking scene requirements)
- The second term is the **switching cost** (flicker/instability penalty)

**Key constraint**: We only know contexts `y_1:t` when choosing action `x_t` (no future information).

## Project Structure

```
Online-Convex-Optimization/
├── main.py                 # Main execution script (vision tuning)
├── robd_algorithm.py       # R-OBD algorithm implementation
├── ml_model.py            # LSTM-based ML predictor
├── hybrid_algorithm.py    # Hybrid algorithms (expert-calibrated learning)
├── data_processor.py      # Vision data loading, normalization, sequence generation
├── evaluation.py          # Performance evaluation and visualization
├── requirements.txt       # Python dependencies
└── results/              # Output directory for results
```

## Features Implemented

### 1. R-OBD Algorithm (`robd_algorithm.py`)
- **Regularized Online Balanced Descent** with optimal hyperparameters
- Implements the theoretical optimal values
- Handles both hitting cost and switching cost regularization
- Provides both single-step prediction and sequence optimization

### 2. ML Model (`ml_model.py`)
- **LSTM-based predictor** for action sequences
- Input: historical actions and contexts
- Output: predicted next action
- Training with OCO-specific loss function
- Configurable architecture (hidden dimensions, layers, dropout)

### 3. Hybrid Algorithms (`hybrid_algorithm.py`)
- **Expert-Calibrated Learning**: Combines ML and R-OBD with adaptive weights
- **Adaptive Hybrid**: Dynamic weight adjustment based on performance
- Shows trade-off between average and worst-case performance

### 4. Data Processing (`data_processor.py`)
- Loads `vision_signals.csv` (or uses synthetic scene signals)
- Scene signals include `scene_brightness` and `feature_density` proxies
- Normalizes context to [0, 1]; 24-step sequences with step size 2
- Train/test split and PyTorch DataLoader integration

### 5. Evaluation (`evaluation.py`)
- OCO cost metrics and vision-centric stability metrics
- Adds flicker proxies: average and peak action deltas
- Cost distribution, action sequences, hybrid weight evolution

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Online-Convex-Optimization
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create results directory**:
   ```bash
   mkdir results
   ```

## Usage

### Quick Start

Run the complete pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess data (synthetic if `vision_signals.csv` not found)
2. Train the ML model
3. Run all algorithms on test data
4. Generate performance comparisons and visualizations
5. Save results to `results/` directory

### Individual Components

#### Test R-OBD Algorithm
```python
from robd_algorithm import ROBDAgent

agent = ROBDAgent(m=5.0)
contexts = [0.2, 0.5, 0.8, 0.3, 0.7]
actions, total_cost = agent.get_total_cost(contexts)
print(f"R-OBD Total Cost: {total_cost}")
```

#### Train ML Model
```python
from ml_model import MLOCOPredictor
from data_processor import DataProcessor

# Load and preprocess data
processor = DataProcessor()
data = processor.load_data('AI workload.csv')
normalized_data = processor.normalize_data(data)
sequences = processor.generate_sequences()
train_sequences, test_sequences = processor.split_data()

# Create data loaders
train_loader, test_loader = processor.create_data_loaders()

# Train ML model
ml_predictor = MLOCOPredictor()
train_losses = ml_predictor.train(train_loader, num_epochs=100)
```

#### Run Hybrid Algorithm
```python
from hybrid_algorithm import ExpertCalibratedLearning

# Initialize components
robd_agent = ROBDAgent()
ml_predictor = MLOCOPredictor()

# Create hybrid algorithm
hybrid_alg = ExpertCalibratedLearning(ml_predictor, robd_agent)

# Run on test sequence
contexts = [0.2, 0.5, 0.8, 0.3, 0.7]
actions, results = hybrid_alg.run_sequence(contexts)
print(f"Hybrid Total Cost: {results['total_cost']}")
```

## Algorithm Details

### R-OBD Algorithm (Flicker-Aware)

The Regularized Online Balanced Descent algorithm maintains two regularizers:
- **λ₁**: Hitting cost regularization (optimal value: 1.0)
- **λ₂**: Switching cost regularization (optimal value: 1.0)

Update rule:
```
x_t = ((m + λ₁) * y_t + λ₂ * x_{t-1}) / (m + λ₁ + λ₂)
```

### ML Model Architecture (Context→Action)

- **Input**: [previous_action, current_context]
- **LSTM**: 2 layers, 64 hidden units
- **Output**: Single action prediction
- **Loss**: OCO-specific cost function
- **Training**: Adam optimizer, learning rate 0.001

### Hybrid Approaches

1. **Expert-Calibrated Learning**:
   - Combines ML and R-OBD predictions
   - Adaptive weight updates based on performance
   - Exponential weights algorithm

2. **Adaptive Hybrid**:
   - Dynamic weight adjustment
   - Performance-based adaptation
   - Windowed evaluation

## Results and Visualizations (Cost + Flicker)

The implementation generates several outputs:

1. **Cost Comparison**: Bar charts comparing total and average costs
2. **Action Sequences**: Time series plots showing contexts vs actions
3. **Weight Evolution**: Tracking how hybrid algorithms adapt weights
4. **Performance Report**: Detailed statistical analysis

## Expected Performance

Based on the implementation:
- **R-OBD**: Provides theoretical guarantees, good worst-case performance
- **ML Model**: Should outperform R-OBD on average (if trained well)
- **Hybrid**: Combines benefits of both approaches

## Complexity Analysis

- **R-OBD (per step)**:
  - Compute update: O(1) in this 1D toy setup; O(d) in d-dim.
  - Sequence of length T: O(T) (or O(T·d)). Memory: O(1) for streaming, O(T) if storing actions.

- **ML Training (LSTM)**:
  - Forward/Backward per batch: O(B · L · H^2) roughly, where B=batch size, L=sequence length (24), H=hidden size (64). Over E epochs and N sequences: O(E · (N/B) · B · L · H^2) ≈ O(E · N · L · H^2).
  - Inference per step: O(H^2). Sequence of length T: O(T · H^2).

- **Expert-Calibrated Hybrid**:
  - Per step prediction combines two experts: O(1) on top of experts.
  - Weight update (exp-weights): O(1). Overall: dominated by chosen expert costs.

- **Data Processing**:
  - Normalization: O(N).
  - Sliding windows (window=24, step=2): O(N) to generate ~O(N) sequences.
  - Data loading to tensors is linear in total windowed samples.

- **Space Complexity**:
  - R-OBD: O(1) streaming; O(T) if storing full action/history.
  - ML model parameters: O(H^2 · L) dominated by LSTM weights; activations during training: O(B · L · H).
  - Datasets: proportional to number of windows times window length.

## File Descriptions

- `main.py`: Complete pipeline execution
- `robd_algorithm.py`: R-OBD implementation with optimal hyperparameters
- `ml_model.py`: LSTM-based ML predictor with OCO loss
- `hybrid_algorithm.py`: Expert-calibrated learning algorithms
- `data_processor.py`: Data loading, normalization, and sequence generation
- `evaluation.py`: Performance evaluation and visualization tools

## Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Additional ML utilities

## References

[1] Gautam Goel, Yiheng Lin, Haoyuan Sun, and Adam Wierman. Beyond online balanced descent: An optimal algorithm for smoothed online optimization. Advances in Neural Information Processing Systems, 32:1875–1885, 2019.

[2] Niangjun Chen, Gautam Goel, and Adam Wierman. Smoothed online convex optimization in high dimensions via online balanced descent. In COLT, 2018.

[3] Guanya Shi, Yiheng Lin, Soon-Jo Chung, Yisong Yue, and Adam Wierman. Online optimization with memory and competitive control. Advances in Neural Information Processing Systems, 33:20636–20647, 2020.

[4] Weici Pan, Guanya Shi, Yiheng Lin, and Adam Wierman. Online optimization with feedback delay and nonlinear switching cost. Proc. ACM Meas. Anal. Comput. Syst., 6(1), Feb 2022.

[5] Pengfei Li, Jianyi Yang, and Shaolei Ren. Expert-calibrated learning for online optimization with switching costs. Proc. ACM Meas. Anal. Comput. Syst., 6(2), Jun 2022.

## License

This project is licensed under the MIT License - see the LICENSE file for details.