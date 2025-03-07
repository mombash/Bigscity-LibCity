import torch
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from mamba_ssm import Mamba as MambaSSM

class MemoryEfficientMamba(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))
        
        # Get data features
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        
        # Get model configs
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.d_model = config.get('d_model', 64)
        self.d_state = config.get('d_state', 16)
        self.d_conv = config.get('d_conv', 4)
        self.expand = config.get('expand', 2)
        self.sub_batch_size = config.get('sub_batch_size', 16)
        
        # Input and output projections
        self.input_proj = nn.Linear(self.num_nodes * self.feature_dim, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.num_nodes * self.output_dim)
        
        # Mamba SSM backbone
        self.mamba = MambaSSM(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
        )
        
        self._logger.info('Memory Efficient Mamba model initialized')
        
    def forward(self, batch):
        """
        Args:
            batch: a dict containing
                X (torch.Tensor): input data with shape [batch_size, input_window, num_nodes, feature_dim]
                y (torch.Tensor): labels with shape [batch_size, output_window, num_nodes, output_dim]
        Returns:
            torch.Tensor: outputs with shape [batch_size, output_window, num_nodes, output_dim]
        """
        # X shape: [batch_size, input_window, num_nodes, feature_dim]
        X = batch['X']
        batch_size = X.shape[0]
        
        # Process in smaller batches to save memory
        batch_outputs = []
        
        for i in range(0, batch_size, self.sub_batch_size):
            # Clear GPU cache at the start of each sub-batch
            torch.cuda.empty_cache()
            
            sub_batch = X[i:i+self.sub_batch_size]
            sub_batch_size = sub_batch.shape[0]
            
            # Reshape input: [sub_batch_size, input_window, num_nodes * feature_dim]
            X_flat = sub_batch.reshape(sub_batch_size, self.input_window, -1)
            
            # Project to d_model dimension
            X_proj = self.input_proj(X_flat)  # [sub_batch_size, input_window, d_model]
            
            # Pass through Mamba backbone
            mamba_out = self.mamba(X_proj)  # [sub_batch_size, input_window, d_model]
            
            # Generate predictions for each step in output window
            sub_outputs = []
            current_input = mamba_out[:, -1:, :]  # Take the last timestep's output
            
            for t in range(self.output_window):
                # Project to output dimension and reshape
                current_pred = self.output_proj(current_input[:, -1, :])
                current_pred = current_pred.reshape(sub_batch_size, 1, self.num_nodes, self.output_dim)
                sub_outputs.append(current_pred)
                
                if t < self.output_window - 1:
                    # Use the prediction as input for the next timestep
                    next_input = current_pred.reshape(sub_batch_size, 1, -1)
                    next_input = self.input_proj(next_input)
                    
                    # Free memory
                    del current_input
                    torch.cuda.empty_cache()
                    
                    current_input = self.mamba(torch.cat([mamba_out[:, -1:, :], next_input], dim=1))[:, -1:, :]
            
            # Combine predictions for this sub-batch
            sub_outputs = torch.cat(sub_outputs, dim=1)  # [sub_batch_size, output_window, num_nodes, output_dim]
            batch_outputs.append(sub_outputs)
            
            # Clear sub-batch tensors
            del sub_batch, X_flat, X_proj, mamba_out, sub_outputs
            torch.cuda.empty_cache()
        
        # Combine all sub-batches
        outputs = torch.cat(batch_outputs, dim=0)  # [batch_size, output_window, num_nodes, output_dim]
        return outputs
    
    def predict(self, batch):
        """
        Predict without teaching forcing
        Args:
            batch: a dict containing
                X (torch.Tensor): input data with shape [batch_size, input_window, num_nodes, feature_dim]
        Returns:
            torch.Tensor: outputs with shape [batch_size, output_window, num_nodes, output_dim]
        """
        return self.forward(batch)
    
    def calculate_loss(self, batch):
        """
        Calculate the loss
        Args:
            batch: a dict containing
                X (torch.Tensor): input data with shape [batch_size, input_window, num_nodes, feature_dim]
                y (torch.Tensor): labels with shape [batch_size, output_window, num_nodes, output_dim]
        Returns:
            torch.Tensor: loss
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0) 