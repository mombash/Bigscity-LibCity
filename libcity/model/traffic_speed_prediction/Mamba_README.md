# Enhanced Mamba Model for Traffic Speed Prediction

This document provides a guide to the enhanced Mamba model for traffic speed prediction, featuring dual temporal/spatial processing paths and advanced embeddings.

## TL;DR: Quick Start Guide

To quickly run the enhanced Mamba model:

1.  **Navigate to the LibCity root directory**:
    ```bash
    cd path/to/Bigscity-LibCity
    ```

2.  **Run the model with the default configuration**:
    ```bash
    python run_model.py --task traffic_state_pred --model Mamba --dataset PEMSD8
    ```
    *This uses the configuration in `libcity/config/model/traffic_state_pred/Mamba.json`, which includes settings for embeddings.*

3.  **View results**: Check the output directory for evaluation metrics and predictions:
    ```bash
    cd libcity/cache/Mamba_PEMSD8_*/evaluate_cache/
    ```

## Detailed Guide: Running with Custom Parameters

### Configuration Options

Customize the model via its configuration file or command-line arguments:

1.  **Edit the configuration file**:
    -   Located at: `libcity/config/model/traffic_state_pred/Mamba.json`
    -   Key parameters include Mamba settings (`d_model`, `d_state`, `d_conv`, `expand`, `num_layers`), training parameters (`max_epoch`, `learning_rate`), and **embedding dimensions** (`input_embedding_dim`, `tod_embedding_dim`, `dow_embedding_dim`, `spatial_embedding_dim`, `adaptive_embedding_dim`).

    ```json
    {
      "d_model": 96,        // Hidden dimension size for Mamba blocks
      "d_state": 32,        // State size in the SSM
      "d_conv": 4,          // Kernel size for the local convolution
      "expand": 2,          // Expansion factor for the hidden dimension
      "num_layers": 3,      // Number of stacked EnhancedMambaLayers per path
      "dropout": 0.1,       // Dropout rate
      "max_epoch": 100,     // Maximum training epochs
      "batch_size": 64,
      "learner": "adam",
      "learning_rate": 0.001,
      "input_window": 12,   // Number of time steps for input
      "output_window": 12,  // Number of time steps to predict
      "executor": "MambaExecutor",
      // Embedding Configs (STAEformer-style)
      "input_embedding_dim": 24,
      "tod_embedding_dim": 24,
      "dow_embedding_dim": 24,
      "spatial_embedding_dim": 16,
      "adaptive_embedding_dim": 80,
      "add_time_in_day": true,   // Enable time-of-day embeddings
      "add_day_in_week": true,   // Enable day-of-week embeddings
      "steps_per_day": 288      // Needed for time-of-day embedding size
      // ... other training parameters
    }
    ```

2.  **Run with a custom configuration file**:
    ```bash
    python run_model.py --task traffic_state_pred --model Mamba --dataset PEMSD8 --config_file path/to/your/custom_mamba_config.json
    ```

3.  **Pass parameters directly via command line**:
    ```bash
    python run_model.py --task traffic_state_pred --model Mamba --dataset PEMSD8 --d_model 128 --d_state 64 --num_layers 4 --dropout 0.15
    ```

## Comprehensive Architecture Review: Enhanced Mamba

The enhanced Mamba model integrates STAEformer-style embeddings and uses parallel processing paths for temporal and spatial information.

### Core Components

#### 1. STAEformer-style Embeddings

The model first processes input features using various embeddings:

```python
# In __init__
self.input_proj = nn.Linear(self.feature_dim, self.input_embedding_dim)
if self.add_time_in_day:
    self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
if self.add_day_in_week:
    self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
self.spatial_embedding = nn.Parameter(...) # Learnable spatial embedding per node
if self.adaptive_embedding_dim > 0:
    self.adaptive_embedding = nn.Parameter(...) # Learnable adaptive embedding per time-step/node

# In forward
features = []
x_main = self.input_proj(x)
features.append(x_main)
# Append tod_emb, dow_emb, spatial_emb, adp_emb if enabled
x = torch.cat(features, dim=-1) # Concatenate all embeddings
```

-   **Input Embedding**: Projects raw features (`feature_dim`) to `input_embedding_dim`.
-   **Time Embeddings**: Optional embeddings for time-of-day (`tod_embedding`) and day-of-week (`dow_embedding`).
-   **Spatial Embedding**: Learnable vector for each node, capturing static spatial characteristics.
-   **Adaptive Embedding**: Learnable vector for each input time step and node, capturing dynamic spatiotemporal context.
-   **Concatenation**: All enabled embeddings are concatenated to form the initial representation (`model_dim`).

#### 2. Projection to Mamba Dimension

The concatenated embeddings (`model_dim`) are projected to the Mamba working dimension (`d_model`):

```python
self.mamba_input_proj = nn.Linear(self.model_dim, self.d_model)
x = self.mamba_input_proj(x)
```

#### 3. Dual Processing Paths (Temporal & Spatial)

The model processes the data through two parallel sets of `EnhancedMambaLayer` stacks:

```python
# In __init__
self.temporal_layers = nn.ModuleList([...]) # Stack of EnhancedMambaLayer
self.spatial_layers = nn.ModuleList([...])  # Another stack of EnhancedMambaLayer

# In forward
# Temporal Path (processes each node's time series independently)
x_temporal = x.permute(...) # Reshape: [num_nodes, batch*time, d_model]
for layer in self.temporal_layers:
    x_temporal = layer(x_temporal)
x_temporal = x_temporal.reshape(...) # Reshape back

# Spatial Path (processes each time step's spatial graph independently)
x_spatial = x.permute(...) # Reshape: [time, batch*num_nodes, d_model]
for layer in self.spatial_layers:
    x_spatial = layer(x_spatial)
x_spatial = x_spatial.reshape(...) # Reshape back
```

-   **Temporal Path**: Focuses on patterns *over time* for each sensor independently.
-   **Spatial Path**: Focuses on patterns *across sensors* at each time step independently.

#### 4. EnhancedMambaLayer

Each layer in the temporal and spatial paths is an `EnhancedMambaLayer`:

```python
# Inside EnhancedMambaLayer
self.mamba1 = MambaSSM(...)
self.mamba2 = MambaSSM(...)
self.feed_forward = nn.Sequential(...)
# Forward pass includes LayerNorm, Dropout, and Residual Connections for each block
```

-   **Structure**: Contains two Mamba blocks and a feed-forward network, each preceded by Layer Normalization and followed by Dropout and a residual connection.
-   **Purpose**: Allows for deeper feature extraction within each processing path compared to a single Mamba block.

#### 5. Weighted Combination

The outputs from the temporal and spatial paths are combined using learnable weights:

```python
self.combine_weights = nn.Parameter(torch.randn(2, self.d_model))
x_combined = (x_temporal.permute(...) * self.combine_weights[0] +
              x_spatial.permute(...) * self.combine_weights[1])
```

-   **Mechanism**: A weighted sum where the model learns the importance of temporal vs. spatial features.

#### 6. Final Output Processing

The combined representation is passed through a final Layer Normalization and an output projection:

```python
self.final_layer_norm = nn.LayerNorm(self.d_model)
self.output_proj = nn.Linear(self.d_model, self.output_dim)

x_out = self.final_layer_norm(x_combined)
x_out = self.output_proj(x_out)
```

-   **Normalization**: Stabilizes the final combined features.
-   **Projection**: Maps the internal `d_model` dimension to the desired `output_dim`.
-   **Window Selection**: The final output selects only the required `output_window` time steps.

### Data Flow Summary

1.  Input `[batch, time, nodes, features]`
2.  Generate embeddings (Input, Time, Spatial, Adaptive).
3.  Concatenate embeddings `[batch, time, nodes, model_dim]`.
4.  Project to Mamba dimension `[batch, time, nodes, d_model]`.
5.  **Parallel Paths**:
    -   Temporal Path: Process `[nodes, batch*time, d_model]` through `temporal_layers`.
    -   Spatial Path: Process `[time, batch*nodes, d_model]` through `spatial_layers`.
6.  Combine temporal and spatial outputs using learned weights `[batch, time, nodes, d_model]`.
7.  Final Layer Normalization.
8.  Project to output dimension `[batch, time, nodes, output_dim]`.
9.  Select prediction window `[batch, output_window, nodes, output_dim]`.

### Loss Function

Uses masked Mean Absolute Error (MAE) loss, suitable for traffic data which may contain missing values:

```python
loss.masked_mae_torch(y_predicted, y_true, 0)
```

### Design Philosophy

-   **Hybrid Approach**: Combines the sequence modeling strengths of Mamba with rich, context-aware embeddings inspired by STAEformer.
-   **Parallel Processing**: Explicitly models temporal and spatial dependencies in separate pathways.
-   **Adaptive Combination**: Learns to weigh the importance of temporal vs. spatial information dynamically.
-   **Enhanced Blocks**: Uses a deeper Mamba block (`EnhancedMambaLayer`) for potentially better feature extraction within each path.

This enhanced architecture aims to capture complex spatiotemporal dynamics effectively by leveraging specialized embeddings and parallel processing streams. 