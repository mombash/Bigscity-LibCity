# MambaCoder Model for Traffic Speed Prediction

This document provides a comprehensive guide to the MambaCoder model, which uses a series of encoder-Mamba-decoder blocks for traffic speed prediction.

## TL;DR: Quick Start Guide

To quickly run the MambaCoder model and see results:

1. **Navigate to the LibCity root directory**:
   ```bash
   cd path/to/Bigscity-LibCity
   ```

2. **Run the model with default configuration**:
   ```bash
   python run_model.py --task traffic_state_pred --model MambaCoder --dataset PEMSD8
   ```

3. **View results**: Check the output directory for evaluation metrics and predictions:
   ```bash
   cd libcity/cache/MambaCoder_PEMSD8_*/evaluate_cache/
   ```

## Detailed Guide: Running with Custom Parameters

### Configuration Options

You can customize the MambaCoder model by modifying its configuration file or passing parameters via command line:

1. **Edit the configuration file**: 
   - Located at: `libcity/config/model/traffic_state_pred/MambaCoder.json`
   - Key parameters:
     ```json
     {
       "d_model": 16,       // Hidden dimension size for Mamba
       "d_state": 16,       // State size in the SSM
       "d_conv": 4,         // Kernel size for the local convolution
       "expand": 2,         // Expansion factor for the hidden dimension
       "num_layers": 10,    // Number of stacked encoder-Mamba-decoder blocks
       "hidden_dim": 64,    // Hidden dimension in encoder and decoder
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
   python run_model.py --task traffic_state_pred --model MambaCoder --dataset PEMSD8 --config_file custom_mambacoder_config.json
   ```

3. **Pass parameters directly**:
   ```bash
   python run_model.py --task traffic_state_pred --model MambaCoder --dataset PEMSD8 --d_model 64 --d_state 32 --hidden_dim 128
   ```

### Advanced Customization

For more advanced customization:

1. **Adjust the model architecture**:
   - Change the number of layers:
     ```bash
     python run_model.py --task traffic_state_pred --model MambaCoder --dataset PEMSD8 --num_layers 5
     ```
   - Modify the hidden dimension of encoder/decoder:
     ```bash
     python run_model.py --task traffic_state_pred --model MambaCoder --dataset PEMSD8 --hidden_dim 128
     ```

2. **Experiment with different model sizes**:
   - For better accuracy (but slower training):
     ```bash
     python run_model.py --task traffic_state_pred --model MambaCoder --dataset PEMSD8 --d_model 64 --d_state 32 --hidden_dim 256 --num_layers 5
     ```
   - For faster training (but potentially lower accuracy):
     ```bash
     python run_model.py --task traffic_state_pred --model MambaCoder --dataset PEMSD8 --d_model 32 --d_state 16 --hidden_dim 64 --num_layers 3
     ```

3. **Dataset and training customization**:
   - Adjust input and output windows:
     ```bash
     python run_model.py --task traffic_state_pred --model MambaCoder --dataset PEMSD8 --input_window 24 --output_window 12
     ```
   - Modify training parameters:
     ```bash
     python run_model.py --task traffic_state_pred --model MambaCoder --dataset PEMSD8 --batch_size 64 --max_epoch 150 --learning_rate 0.0005
     ```

## Comprehensive Architecture Review

The MambaCoder model is designed with a unique encoder-Mamba-decoder structure for traffic speed prediction. Let's break down each component:

### Core Components

#### 1. EncoderMambaDecoderBlock

Each block consists of an encoder, a Mamba model, and a decoder:

```python
class EncoderMambaDecoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, d_model, d_state, d_conv, expand):
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        
        # Mamba
        self.mamba = MambaSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
```

**What it does**: Processes input data through three stages: encoding, Mamba processing, and decoding.

**Why it's added**: This structure allows the model to transform the input data into a suitable representation for Mamba processing and then transform it back to the output space.

**How it works**:
- **Encoder**: Maps input data to a higher-dimensional space using two linear layers with a ReLU activation in between.
- **Mamba**: Processes the encoded data using a state space model for efficient sequence modeling.
- **Decoder**: Maps the Mamba output back to the desired output space using two linear layers with a ReLU activation.

#### 2. State Space Model (MambaSSM)

The core of each block is the Mamba SSM:

```python
self.mamba = MambaSSM(
    d_model=d_model,
    d_state=d_state,
    d_conv=d_conv,
    expand=expand
)
```

**What it does**: Processes sequential data using an efficient state space formulation.

**Why it's added**: Provides an efficient alternative to attention mechanisms while maintaining the ability to model long-range dependencies.

**How it works**:
- Models a continuous-time linear system using a discretized state space approach
- Uses selective scanning to reduce computational complexity
- The internal mechanism combines convolution and state updates for efficient sequence processing

#### 3. Stacked Architecture

Multiple EncoderMambaDecoderBlocks are stacked for hierarchical processing:

```python
# Create multiple EncoderMambaDecoder blocks
self.blocks = nn.ModuleList()

if self.num_layers == 1:
    # Single block case
    self.blocks.append(EncoderMambaDecoderBlock(...))
else:
    # First block: input_dim -> intermediate_dim
    self.blocks.append(EncoderMambaDecoderBlock(...))
    
    # Middle blocks: intermediate_dim -> intermediate_dim
    for _ in range(self.num_layers - 2):
        self.blocks.append(EncoderMambaDecoderBlock(...))
    
    # Last block: intermediate_dim -> final_output_dim
    self.blocks.append(EncoderMambaDecoderBlock(...))
```

**What it does**: Processes data through multiple blocks in sequence, with intermediate dimensions between blocks.

**Why it's added**: Allows the model to learn increasingly complex representations through multiple transformations.

**How it works**:
- First block transforms input data to an intermediate representation
- Middle blocks process the intermediate representation
- Last block transforms the intermediate representation to the final output

### Data Flow

The model transforms input data through several steps:

1. **Input Flattening**:
   ```python
   x = x.reshape(batch_size, -1)  # [batch_size, input_window * num_nodes * feature_dim]
   ```
   This flattens the input data for processing by the first encoder.

2. **Block Processing**:
   ```python
   for i, block in enumerate(self.blocks):
       x = block(x)
   ```
   Each block processes the data in sequence, with the output of one block becoming the input to the next.

3. **Output Reshaping**:
   ```python
   return x.reshape(batch_size, self.output_window, self.num_nodes, self.output_dim)
   ```
   The final output is reshaped to the expected format for traffic prediction.

### Input and Output Dimensions

The model handles dimensions carefully:

```python
# Input and output dimensions for each block
input_dim = self.input_window * self.num_nodes * self.feature_dim

# Final output dimension
final_output_dim = self.output_window * self.num_nodes * self.output_dim
```

**What it does**: Calculates the appropriate dimensions for the input and output of each block.

**Why it's added**: Ensures that the model can process the flattened input data and produce the correctly shaped output.

**How it works**:
- Input dimension is the product of input window size, number of nodes, and feature dimension
- Output dimension is the product of output window size, number of nodes, and output dimension

### Loss Function

The model uses masked MAE loss for training:

```python
loss.masked_mae_torch(y_predicted, y_true, 0)
```

**What it does**: Calculates the Mean Absolute Error while ignoring masked values.

**Why it's added**: Traffic data often contains missing values that should be excluded from the loss calculation.

**How it works**: Computes the absolute difference between predictions and ground truth, applies a mask to ignore certain values, and takes the mean.

### Special Handling for Dimensions

The model includes special handling for various input dimensions:

```python
# Process with Mamba (maintaining sequence dimension)
if len(x_encoded.shape) == 2:  # For 2D input, add sequence dimension
    x_encoded = x_encoded.unsqueeze(1)
    
x_mamba = self.mamba(x_encoded)

# Remove sequence dimension if it was added
if len(x.shape) == 2 and len(x_mamba.shape) == 3:
    x_mamba = x_mamba.squeeze(1)
```

**What it does**: Adds or removes sequence dimensions as needed for Mamba processing.

**Why it's added**: Mamba requires a sequence dimension, but the data may not always have one explicitly.

**How it works**: Uses tensor operations to add or remove dimensions without changing the actual data values.

### Overall Design Philosophy

The MambaCoder model is designed with a unique approach to traffic prediction:

1. **Hierarchical Processing**: Multiple blocks process data in sequence for increasingly complex representations.
2. **Flexible Dimensions**: The model can handle various input and output dimensions through careful reshaping.
3. **Efficient Sequence Modeling**: The Mamba SSM provides an efficient alternative to attention for modeling traffic sequences.
4. **Deep Transformation**: The encoder-decoder structure within each block allows for non-linear transformations of the data.

This architecture makes it particularly suitable for complex traffic prediction tasks where both temporal and spatial patterns need to be captured. 