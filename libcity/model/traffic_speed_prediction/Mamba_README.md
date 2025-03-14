# Mamba Model for Traffic Speed Prediction

This document provides a comprehensive guide to the Mamba model for traffic speed prediction using state space models (SSMs).

## TL;DR: Quick Start Guide

To quickly run the Mamba model and see results:

1. **Navigate to the LibCity root directory**:
   ```bash
   cd path/to/Bigscity-LibCity
   ```

2. **Run the model with default configuration**:
   ```bash
   python run_model.py --task traffic_state_pred --model Mamba --dataset PEMSD8
   ```

3. **View results**: Check the output directory for evaluation metrics and predictions:
   ```bash
   cd libcity/cache/Mamba_PEMSD8_*/evaluate_cache/
   ```

## Detailed Guide: Running with Custom Parameters

### Configuration Options

You can customize the Mamba model by modifying its configuration file or passing parameters via command line:

1. **Edit the configuration file**: 
   - Located at: `libcity/config/model/traffic_state_pred/Mamba.json`
   - Key parameters:
     ```json
     {
       "d_model": 16,       // Hidden dimension size of the model
       "d_state": 16,       // State size in the SSM
       "d_conv": 4,         // Kernel size for the local convolution
       "expand": 2,         // Expansion factor for the hidden dimension
       "num_layers": 5,     // Number of stacked Mamba layers
       "max_epoch": 100,    // Maximum training epochs
       "learner": "adam",   // Optimizer
       "learning_rate": 0.001, // Learning rate
       "input_window": 12,  // Number of time steps for input
       "output_window": 12, // Number of time steps to predict
       "executor": "MambaExecutor" // Executor class
     }
     ```

2. **Run with custom configuration file**:
   ```bash
   python run_model.py --task traffic_state_pred --model Mamba --dataset PEMSD8 --config_file custom_mamba_config.json
   ```

3. **Pass parameters directly**:
   ```bash
   python run_model.py --task traffic_state_pred --model Mamba --dataset PEMSD8 --d_model 64 --d_state 32 --num_layers 8
   ```

### Advanced Customization

For more advanced customization:

1. **Create a custom dataset configuration**:
   - Located at: `libcity/config/data/PEMSD8.json`
   - Adjust preprocessing parameters like data splitting and normalization

2. **Modify the training process**:
   - Batch size, learning rate, and training schedules can be adjusted
   - Example:
     ```bash
     python run_model.py --task traffic_state_pred --model Mamba --dataset PEMSD8 --batch_size 64 --learning_rate 0.0005 --max_epoch 150
     ```

3. **Experiment with different model sizes**:
   - For better accuracy (but slower training):
     ```bash
     python run_model.py --task traffic_state_pred --model Mamba --dataset PEMSD8 --d_model 96 --d_state 32 --num_layers 8
     ```
   - For faster training (but potentially lower accuracy):
     ```bash
     python run_model.py --task traffic_state_pred --model Mamba --dataset PEMSD8 --d_model 32 --d_state 16 --num_layers 3
     ```

## Comprehensive Architecture Review

The Mamba model is designed for spatiotemporal traffic speed prediction using state space models (SSMs). Let's break down each component:

### Core Components

#### 1. State Space Model (MambaSSM)

The heart of the model is the Mamba SSM (State Space Model), which processes sequential data efficiently:

```python
self.mamba_layers = nn.ModuleList([
    MambaSSM(
        d_model=self.d_model,
        d_state=self.d_state,
        d_conv=self.d_conv,
        expand=self.expand
    ) for _ in range(self.num_layers)
])
```

**What it does**: Processes temporal patterns in the data using an efficient state space formulation.

**Why it's added**: Mamba provides a more efficient alternative to traditional attention mechanisms while maintaining the ability to model long-range dependencies.

**How it works**: 
- Models a continuous-time linear system using a discretized state space approach
- Uses selective scanning to reduce computational complexity
- The internal mechanism uses a combination of convolution and state updates that allow the model to selectively remember or forget information

#### 2. Layer Normalization

Layer normalization is applied after each Mamba layer:

```python
self.layer_norms = nn.ModuleList([
    nn.LayerNorm(self.d_model) for _ in range(self.num_layers)
])
```

**What it does**: Normalizes the outputs of each layer to improve training stability.

**Why it's added**: Helps with gradient flow and speeds up convergence by normalizing activations.

**How it works**: Normalizes the features across the feature dimension for each sample independently, maintaining a mean of 0 and standard deviation of 1.

#### 3. Input and Output Projections

The model includes projections to transform between feature spaces:

```python
self.input_proj = nn.Linear(self.feature_dim, self.d_model)
self.output_proj = nn.Linear(self.d_model, self.output_dim)
```

**What it does**: Maps the input features to the model's internal dimension and transforms the model outputs back to the prediction space.

**Why it's added**: The model works with a fixed internal dimension that may differ from the input/output dimensions.

**How it works**: Simple linear transformations (matrix multiplications) to change the dimensionality of the data.

#### 4. Residual Connections

Residual connections are implemented in the forward pass:

```python
if i > 0:
    x = x + mamba_output
else:
    x = mamba_output
```

**What it does**: Creates shortcuts that allow gradients to flow more easily through the network.

**Why it's added**: Helps combat the vanishing gradient problem in deep networks and improves training.

**How it works**: Adds the input of a layer directly to its output, creating a path for gradients to bypass layers.

### Data Flow

The model transforms input data through several steps:

1. **Input Reshaping**: 
   ```python
   x = x.permute(0, 2, 1, 3).contiguous()
   x = x.reshape(batch_size * self.num_nodes, self.input_window, self.feature_dim)
   ```
   This preserves spatial structure by processing each node separately.

2. **Feature Projection**:
   ```python
   x = self.input_proj(x)
   ```
   Maps input features to model dimensions.

3. **Sequential Processing**:
   The data flows through multiple Mamba layers with layer normalization and residual connections.

4. **Output Generation**:
   ```python
   x = self.output_proj(x)
   x = x[:, -self.output_window:, :]
   ```
   Projects to output dimension and selects the prediction window.

5. **Output Reshaping**:
   ```python
   x = x.reshape(batch_size, self.num_nodes, self.output_window, self.output_dim)
   x = x.permute(0, 2, 1, 3).contiguous()
   ```
   Transforms back to the expected output format.

### Key Innovations

1. **Node-wise Processing**: The model preserves spatial structure by processing each node separately, which helps capture node-specific temporal patterns.

2. **Stacked Architecture**: Multiple Mamba layers are stacked to capture increasingly complex patterns in the data.

3. **Efficient Sequence Modeling**: The Mamba SSM provides an efficient alternative to attention for modeling long sequences.

### Loss Function

The model uses masked MAE loss for training:

```python
loss.masked_mae_torch(y_predicted, y_true, 0)
```

**What it does**: Calculates the Mean Absolute Error while ignoring masked values.

**Why it's added**: Traffic data often contains missing values that should be excluded from loss calculation.

**How it works**: Computes the absolute difference between predictions and ground truth, applies a mask to ignore certain values, and takes the mean.

### Overall Design Philosophy

The Mamba model is designed with efficiency and effectiveness in mind:

1. **Spatiotemporal Modeling**: Preserves both temporal and spatial aspects of traffic data
2. **Scalability**: Efficiently handles long sequences and multiple nodes
3. **Deep Architecture**: Uses multiple layers for hierarchical feature extraction
4. **Stability**: Employs normalization and residual connections for stable training

This architecture makes it well-suited for traffic speed prediction tasks, where capturing both temporal patterns and spatial relationships is crucial. 