import torch
import torch.nn as nn
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

from mamba_ssm import Mamba as MambaSSM


class EncoderMambaDecoderBlock(nn.Module):
    """A block consisting of an encoder, a Mamba model, and a decoder"""
    def __init__(self, input_dim, hidden_dim, output_dim, d_model, d_state, d_conv, expand):
        super().__init__()
        
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
        
    def forward(self, x):
        # Encode
        x_encoded = self.encoder(x)
        
        # Process with Mamba (maintaining sequence dimension)
        if len(x_encoded.shape) == 2:  # For 2D input, add sequence dimension
            x_encoded = x_encoded.unsqueeze(1)
            
        x_mamba = self.mamba(x_encoded)
        
        # Remove sequence dimension if it was added
        if len(x.shape) == 2 and len(x_mamba.shape) == 3:
            x_mamba = x_mamba.squeeze(1)
            
        # Decode
        x_decoded = self.decoder(x_mamba)
        
        return x_decoded


class MambaCoder(AbstractTrafficStateModel):
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
        self._logger.info(f"Building MambaCoder model with {self.num_layers} stacked layers")
        
        # Hidden dimension for encoder/decoder
        self.hidden_dim = config.get('hidden_dim', 64)
        
        # Input and output dimensions for each block
        input_dim = self.input_window * self.num_nodes * self.feature_dim
        
        # Final output dimension
        final_output_dim = self.output_window * self.num_nodes * self.output_dim
        
        # Create multiple EncoderMambaDecoder blocks
        self.blocks = nn.ModuleList()
        
        if self.num_layers == 1:
            # Single block case
            self.blocks.append(
                EncoderMambaDecoderBlock(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=final_output_dim,
                    d_model=self.d_model,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand
                )
            )
        else:
            # First block: input_dim -> intermediate_dim
            intermediate_dim = self.hidden_dim
            self.blocks.append(
                EncoderMambaDecoderBlock(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=intermediate_dim,
                    d_model=self.d_model,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand
                )
            )
            
            # Middle blocks: intermediate_dim -> intermediate_dim
            for _ in range(self.num_layers - 2):
                self.blocks.append(
                    EncoderMambaDecoderBlock(
                        input_dim=intermediate_dim,
                        hidden_dim=self.hidden_dim,
                        output_dim=intermediate_dim,
                        d_model=self.d_model,
                        d_state=self.d_state,
                        d_conv=self.d_conv,
                        expand=self.expand
                    )
                )
            
            # Last block: intermediate_dim -> final_output_dim
            self.blocks.append(
                EncoderMambaDecoderBlock(
                    input_dim=intermediate_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=final_output_dim,
                    d_model=self.d_model,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand
                )
            )
        
        # Log the device being used
        self._logger.info(f"MambaCoder model configured for device: {self.device}")

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
        
        # Flatten the input for the first encoder
        x = x.reshape(batch_size, -1)  # [batch_size, input_window * num_nodes * feature_dim]
        
        # Process through stacked blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
        # Reshape to the expected output format
        return x.reshape(batch_size, self.output_window, self.num_nodes, self.output_dim)

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
        test_tensor = torch.randn(1, self.input_window * self.num_nodes * self.feature_dim)
        test_tensor = test_tensor.to(self.device)
        
        # Check device of model components
        for i, block in enumerate(self.blocks):
            self._logger.info(f"Device of block {i} encoder first parameter: {next(block.encoder.parameters()).device}")
            self._logger.info(f"Device of block {i} mamba first parameter: {next(block.mamba.parameters()).device}")
            self._logger.info(f"Device of block {i} decoder first parameter: {next(block.decoder.parameters()).device}")
        
        self._logger.info(f"Device of test tensor: {test_tensor.device}")
        
        # Try a forward pass with the first block and check output device
        with torch.no_grad():
            self.blocks[0].encoder.eval()
            out = self.blocks[0].encoder(test_tensor)
            self._logger.info(f"Device of output from first block encoder: {out.device}")
        
        # Return True if all components are on the correct device
        return all(p.device == self.device for p in self.parameters()) 