import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

from mamba_ssm import Mamba as MambaSSM


class EnhancedMambaLayer(nn.Module):
    """Enhanced Mamba layer with additional components but no attention"""
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.1):
        super().__init__()
        # First Mamba block with pre-norm
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mamba1 = MambaSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Second Mamba block with pre-norm (different parameters)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.mamba2 = MambaSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward block with pre-norm
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x):
        # First Mamba block with residual
        residual = x
        x = self.layer_norm1(x)
        x = self.mamba1(x)
        x = self.dropout1(x)
        x = x + residual
        
        # Second Mamba block with residual
        residual = x
        x = self.layer_norm2(x)
        x = self.mamba2(x)
        x = self.dropout2(x)
        x = x + residual
        
        # Feed-forward with residual
        residual = x
        x = self.layer_norm3(x)
        x = self.feed_forward(x)
        x = self.dropout3(x)
        x = x + residual
        
        return x


class Mamba(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # Get data features
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        # Get model config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        
        # Get Mamba-specific parameters from config
        self.d_model = config.get('d_model', 96)  # Increased from 16 to 96
        self.d_state = config.get('d_state', 32)  # Increased from 16 to 32
        self.d_conv = config.get('d_conv', 4)
        self.expand = config.get('expand', 2)
        self.dropout = config.get('dropout', 0.1)  # Added dropout
        
        # Number of stacked Mamba layers
        self.num_layers = config.get('num_layers', 3)
        self._logger.info(f"Building Enhanced Mamba model with {self.num_layers} layers")

        # Input embedding and output projection layers
        self.input_proj = nn.Linear(self.feature_dim, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.output_dim)
        
        # Layer normalization for input
        self.input_layer_norm = nn.LayerNorm(self.d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Embedding(self.input_window * max(1, config.get('batch_size', 64)), self.d_model)
        
        # Enhanced Mamba layers
        self.enhanced_layers = nn.ModuleList([
            EnhancedMambaLayer(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Add skip connection adapter
        self.skip_adapter = nn.Linear(self.d_model * self.num_layers, self.d_model)
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(self.d_model)
        
        # Log the device being used
        self._logger.info(f"Enhanced Mamba model configured for device: {self.device}")

    def forward(self, batch):
        """
        Forward pass through the model
        :param batch: Input data dictionary
        :return: Predicted values with shape [batch_size, output_window, num_nodes, output_dim]
        """
        x = batch['X']  # [batch_size, input_window, num_nodes, feature_dim]
        
        # Make sure input is on the correct device
        x = x.to(self.device)
        
        batch_size = x.shape[0]
        
        # Reshape for processing each node across all batches and time steps
        # [batch_size, input_window, num_nodes, feature_dim] -> [num_nodes, batch_size * input_window, feature_dim]
        x = x.permute(2, 0, 1, 3).contiguous()
        x = x.reshape(self.num_nodes, batch_size * self.input_window, self.feature_dim)
        
        # Project input to model dimension
        x = self.input_proj(x)  # [num_nodes, batch_size * input_window, d_model]
        
        # Apply input layer normalization
        x = self.input_layer_norm(x)
        
        # Add positional encoding - need different approach since sequence length is now batch_size * input_window
        positions = torch.arange(0, batch_size * self.input_window, device=self.device).unsqueeze(0).expand(self.num_nodes, -1)
        pos_encoding = self.pos_encoder(positions)
        x = x + pos_encoding
        
        # Store layer outputs for skip connections
        layer_outputs = []
        
        # Process with enhanced Mamba layers
        for layer in self.enhanced_layers:
            x = layer(x)
            layer_outputs.append(x)
        
        # Concatenate all layer outputs for skip connections
        if len(layer_outputs) > 1:
            concatenated = torch.cat(layer_outputs, dim=-1)
            x = self.skip_adapter(concatenated)
        
        # Apply final layer normalization
        x = self.final_layer_norm(x)
        
        # Project output to feature dimension
        x = self.output_proj(x)  # [num_nodes, batch_size * input_window, output_dim]
        
        # Reshape back to separate batch and time dimensions
        x = x.reshape(self.num_nodes, batch_size, self.input_window, self.output_dim)
        
        # Take the last 'output_window' steps for each batch
        if self.input_window >= self.output_window:
            x = x[:, :, -self.output_window:, :]  # [num_nodes, batch_size, output_window, output_dim]
        else:
            # Need to generate predictions beyond input window for each node
            # This is a simplified approach - for proper implementation, you'd need to rework the autoregressive generation
            last_points = x[:, :, -1:, :]  # [num_nodes, batch_size, 1, output_dim]
            extended_predictions = last_points.repeat(1, 1, self.output_window - self.input_window, 1)
            x = torch.cat([x[:, :, :self.input_window, :], extended_predictions], dim=2)  # [num_nodes, batch_size, output_window, output_dim]
        
        # Final reshape to expected output format
        x = x.permute(1, 2, 0, 3).contiguous()  # [batch_size, output_window, num_nodes, output_dim]
        
        return x

    def calculate_loss(self, batch):
        """
        Calculate the training loss for a batch of data
        :param batch: Input data dictionary
        :return: Training loss (tensor)
        """
        y_true = batch['y'].to(self.device)
        y_predicted = self.predict(batch)
        
        # Apply inverse normalization
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        
        # Calculate masked MAE loss
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        """
        Make predictions for a batch of data
        :param batch: Input data dictionary
        :return: Predictions with shape [batch_size, output_window, num_nodes, output_dim]
        """
        return self.forward(batch)

    def check_gpu_usage(self):
        """
        Method to check and log whether the model is actually using the GPU
        """
        # Create a small test tensor
        test_tensor = torch.randn(1, 1, self.feature_dim)
        test_tensor = test_tensor.to(self.device)
        
        # Check device of model components
        self._logger.info(f"Device of input_proj weight: {self.input_proj.weight.device}")
        self._logger.info(f"Device of output_proj weight: {self.output_proj.weight.device}")
        for i, layer in enumerate(self.enhanced_layers):
            self._logger.info(f"Device of enhanced_layer {i} first parameter: {next(layer.parameters()).device}")
        self._logger.info(f"Device of test tensor: {test_tensor.device}")
        
        # Try a forward pass with the test tensor and check output device
        with torch.no_grad():
            self.input_proj.eval()
            out = self.input_proj(test_tensor)
            self._logger.info(f"Device of output from input_proj: {out.device}")
        
        # Return True if all components are on the correct device
        return all(p.device == self.device for p in self.parameters())
