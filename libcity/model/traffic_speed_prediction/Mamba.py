import torch
import torch.nn as nn
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

from mamba_ssm import Mamba as MambaSSM


class Mamba(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # Get data features
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        # Get model config
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        
        # Get Mamba-specific parameters from config
        self.d_model = config.get('d_model', 16)
        self.d_state = config.get('d_state', 16)
        self.d_conv = config.get('d_conv', 4)
        self.expand = config.get('expand', 2)
        
        # Number of stacked Mamba layers
        self.num_layers = config.get('num_layers', 1)
        self._logger.info(f"Building Mamba model with {self.num_layers} layers")

        # Input and output projection layers to match dimensions
        self.input_proj = nn.Linear(self.feature_dim, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.output_dim)
        
        # Create multiple Mamba layers stacked in series
        self.mamba_layers = nn.ModuleList([
            MambaSSM(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand
            ) for _ in range(self.num_layers)
        ])
        
        # Add layer normalization between Mamba layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.num_layers)
        ])
        
        # Log the device being used
        self._logger.info(f"Mamba model configured for device: {self.device}")

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
        
        # Reshape for processing each node and time step
        # [batch_size, input_window, num_nodes, feature_dim] -> [batch_size * num_nodes, input_window, feature_dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size * self.num_nodes, self.input_window, self.feature_dim)
        
        # Project input to model dimension
        x = self.input_proj(x)  # [batch_size * num_nodes, input_window, d_model]
        
        # Process with stacked Mamba layers
        for i, (mamba_layer, layer_norm) in enumerate(zip(self.mamba_layers, self.layer_norms)):
            # Pass through the Mamba layer
            mamba_output = mamba_layer(x)
            
            # Apply layer normalization
            mamba_output = layer_norm(mamba_output)
            
            # Residual connection (for layers after the first one)
            if i > 0:
                x = x + mamba_output
            else:
                x = mamba_output
        
        # Project output to feature dimension
        x = self.output_proj(x)  # [batch_size * num_nodes, input_window, output_dim]
        
        # Take the last 'output_window' steps
        x = x[:, -self.output_window:, :]  # [batch_size * num_nodes, output_window, output_dim]
        
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
        for i, layer in enumerate(self.mamba_layers):
            self._logger.info(f"Device of mamba_layer {i} first parameter: {next(layer.parameters()).device}")
        self._logger.info(f"Device of test tensor: {test_tensor.device}")
        
        # Try a forward pass with the test tensor and check output device
        with torch.no_grad():
            self.input_proj.eval()
            out = self.input_proj(test_tensor)
            self._logger.info(f"Device of output from input_proj: {out.device}")
        
        # Return True if all components are on the correct device
        return all(p.device == self.device for p in self.parameters())
