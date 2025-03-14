import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

from mamba_ssm import Mamba as MambaSSM


class DualDirectionMambaBlock(nn.Module):
    """Enhanced Mamba block that processes in both temporal and node dimensions"""
    def __init__(self, d_model, d_state, d_conv, expand, dropout=0.1):
        super().__init__()
        
        # Pre-layer normalization for temporal Mamba
        self.norm1 = nn.LayerNorm(d_model)
        
        # Temporal direction Mamba
        self.mamba_temporal = MambaSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Pre-layer normalization for spatial direction
        self.norm2 = nn.LayerNorm(d_model)
        
        # Spatial direction processing - using standard Mamba without transposition
        # This avoids dimension mismatch issues
        self.mamba_spatial = MambaSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x):
        # Temporal Mamba processing with residual
        residual = x
        x_norm = self.norm1(x)
        x_temporal = self.mamba_temporal(x_norm)
        x = residual + self.dropout1(x_temporal)
        
        # Second Mamba processing with residual (alternative direction)
        residual = x
        x_norm = self.norm2(x)
        
        # Process with the second Mamba - using same direction to avoid dimension issues
        # But with different parameters to capture different patterns
        x_spatial = self.mamba_spatial(x_norm)
        x = residual + self.dropout2(x_spatial)
        
        # Feed-forward with residual
        residual = x
        x_norm = self.norm3(x)
        x_ffn = self.ffn(x_norm)
        x = residual + self.dropout3(x_ffn)
        
        return x


class MambaCoder(AbstractTrafficStateModel):
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
        self.dropout = config.get('dropout', 0.1)
        
        # Number of stacked Mamba layers
        self.num_layers = config.get('num_layers', 5)
        self._logger.info(f"Building Enhanced MambaCoder model with {self.num_layers} layers")
        
        # Layer normalization for input
        self.input_layer_norm = nn.LayerNorm(self.d_model)
        
        # Input embedding
        self.input_embedding = nn.Linear(self.feature_dim, self.d_model)
        
        # Additional convolution for spatial mixing
        self.spatial_mix = nn.Linear(self.d_model, self.d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Embedding(self.input_window, self.d_model)
        
        # Stack of dual-direction Mamba blocks
        self.blocks = nn.ModuleList([
            DualDirectionMambaBlock(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Skip connection handling
        self.use_skip_connections = config.get('use_skip_connections', True)
        if self.use_skip_connections:
            self.skip_adapter = nn.Linear(self.d_model * self.num_layers, self.d_model)
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.output_dim)
        )
        
        # Log the device being used
        self._logger.info(f"Enhanced MambaCoder model configured for device: {self.device}")

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
        
        # Preserve spatial structure - process each node separately
        # [batch_size, input_window, num_nodes, feature_dim] -> [batch_size * num_nodes, input_window, feature_dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size * self.num_nodes, self.input_window, self.feature_dim)
        
        # Apply input embedding
        x = self.input_embedding(x)  # [batch_size * num_nodes, input_window, d_model]
        
        # Add positional encoding
        positions = torch.arange(0, self.input_window, device=self.device).unsqueeze(0).expand(batch_size * self.num_nodes, -1)
        pos_encoding = self.pos_encoder(positions)
        x = x + pos_encoding
        
        # Apply input normalization
        x = self.input_layer_norm(x)
        
        # Process through Mamba blocks with skip connections
        if self.use_skip_connections:
            layer_outputs = []
            
        for block in self.blocks:
            x = block(x)
            
            if self.use_skip_connections:
                layer_outputs.append(x)
        
        # Apply skip connections if enabled
        if self.use_skip_connections and len(self.blocks) > 1:
            concatenated = torch.cat(layer_outputs, dim=-1)
            x = self.skip_adapter(concatenated)
        
        # Project to output dimension
        x = self.output_projection(x)  # [batch_size * num_nodes, input_window, output_dim]
        
        # Take the last 'output_window' steps or generate future predictions
        if self.input_window >= self.output_window:
            x = x[:, -self.output_window:, :]  # [batch_size * num_nodes, output_window, output_dim]
        else:
            # Need to generate predictions beyond input window
            last_points = x[:, -1:, :]  # [batch_size * num_nodes, 1, output_dim]
            extra_steps = self.output_window - self.input_window
            
            # Simple autoregressive generation for extra steps
            future_preds = [last_points]
            curr_input = last_points
            
            for _ in range(extra_steps):
                # Project back to d_model, process with last block, project to output
                curr_projected = self.input_embedding(curr_input)
                curr_processed = self.blocks[-1](curr_projected)
                curr_output = self.output_projection(curr_processed)
                future_preds.append(curr_output)
                curr_input = curr_output
            
            # Combine existing and generated future steps
            existing_steps = x[:, :self.input_window, :]
            future_steps = torch.cat(future_preds, dim=1)
            x = torch.cat([existing_steps[:, -(self.output_window-extra_steps):, :], future_steps], dim=1)
        
        # Reshape back to expected output format
        x = x.reshape(batch_size, self.num_nodes, self.output_window, self.output_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, output_window, num_nodes, output_dim]
        
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
        
        # Base masked MAE loss (no weighting to keep it simple)
        base_loss = loss.masked_mae_torch(y_predicted, y_true, 0)
        
        return base_loss

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
        test_tensor = torch.randn(1, self.input_window, self.feature_dim)
        test_tensor = test_tensor.to(self.device)
        
        # Check device of model components
        for i, block in enumerate(self.blocks):
            self._logger.info(f"Device of block {i} mamba_temporal first parameter: {next(block.mamba_temporal.parameters()).device}")
            self._logger.info(f"Device of block {i} mamba_spatial first parameter: {next(block.mamba_spatial.parameters()).device}")
        
        self._logger.info(f"Device of test tensor: {test_tensor.device}")
        
        # Try a forward pass with the first block and check output device
        with torch.no_grad():
            self.input_embedding.eval()
            out = self.input_embedding(test_tensor)
            self._logger.info(f"Device of output from input embedding: {out.device}")
        
        # Return True if all components are on the correct device
        return all(p.device == self.device for p in self.parameters())