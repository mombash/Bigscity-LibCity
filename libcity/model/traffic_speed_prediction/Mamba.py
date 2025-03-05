import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
import math

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x

class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """
        :param x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        :return: (batch_size, c_out, input_length-kt+1, num_nodes)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)

class OutputLayer(nn.Module):
    def __init__(self, c, t, n, out_dim):
        super(OutputLayer, self).__init__()
        self.tconv1 = TemporalConvLayer(t, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = TemporalConvLayer(1, c, c, "sigmoid")
        self.fc = nn.Conv2d(c, out_dim, 1)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)

class Mamba(AbstractTrafficStateModel):
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
        self.kt = config.get('kt', 3)  # Temporal kernel size
        self.drop_prob = config.get('dropout', 0.1)
        
        # Define temporal convolution blocks (similar to STGCN)
        # First block: feature_dim -> d_model -> d_model*2
        self.temp_conv1 = TemporalConvLayer(self.kt, self.feature_dim, self.d_model, "GLU")
        self.temp_conv2 = TemporalConvLayer(self.kt, self.d_model, self.d_model*2)
        
        # Layer normalization and dropout
        self.ln1 = nn.LayerNorm([self.num_nodes, self.d_model*2])
        self.dropout1 = nn.Dropout(self.drop_prob)
        
        # Second block: d_model*2 -> d_model -> d_model*2
        self.temp_conv3 = TemporalConvLayer(self.kt, self.d_model*2, self.d_model, "GLU")
        self.temp_conv4 = TemporalConvLayer(self.kt, self.d_model, self.d_model*2)
        
        # Layer normalization and dropout
        self.ln2 = nn.LayerNorm([self.num_nodes, self.d_model*2])
        self.dropout2 = nn.Dropout(self.drop_prob)
        
        # Output layer
        remaining_length = self.input_window - 4 * (self.kt - 1)  # After 4 temporal convolutions
        if remaining_length <= 0:
            raise ValueError(f"Input window too small for kernel size. Need at least {4*(self.kt-1)+1}")
            
        self.output_layer = OutputLayer(
            self.d_model*2, 
            remaining_length, 
            self.num_nodes, 
            self.output_dim
        )
        
        self._logger.info('Modified Mamba model initialized with STGCN temporal blocks')
        
    def forward(self, batch):
        """
        Args:
            batch: a dict containing
                X (torch.Tensor): input data with shape [batch_size, input_window, num_nodes, feature_dim]
        Returns:
            torch.Tensor: outputs with shape [batch_size, output_window, num_nodes, output_dim]
        """
        # X shape: [batch_size, input_window, num_nodes, feature_dim]
        X = batch['X']
        batch_size = X.shape[0]
        
        # Reshape to match STGCN input shape [batch_size, feature_dim, input_window, num_nodes]
        X = X.permute(0, 3, 1, 2)
        
        # First temporal block
        x1 = self.temp_conv1(X)
        x2 = self.temp_conv2(x1)
        x2 = self.ln1(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x2 = self.dropout1(x2)
        
        # Second temporal block
        x3 = self.temp_conv3(x2)
        x4 = self.temp_conv4(x3)
        x4 = self.ln2(x4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x4 = self.dropout2(x4)
        
        # Output layer
        output = self.output_layer(x4)  # [batch_size, output_dim, 1, num_nodes]
        
        # Generate predictions for each step in output window
        outputs = []
        current_input = output.permute(0, 2, 3, 1)  # [batch_size, 1, num_nodes, output_dim]
        outputs.append(current_input)
        
        # Autoregressive generation for multi-step prediction
        for t in range(1, self.output_window):
            # Create new input by appending the latest prediction
            new_x = torch.cat([X[:, :, 1:, :], current_input.permute(0, 3, 1, 2)], dim=2)
            
            # Process through temporal blocks
            x1 = self.temp_conv1(new_x)
            x2 = self.temp_conv2(x1)
            x2 = self.ln1(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x2 = self.dropout1(x2)
            
            x3 = self.temp_conv3(x2)
            x4 = self.temp_conv4(x3)
            x4 = self.ln2(x4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x4 = self.dropout2(x4)
            
            # Get next prediction
            next_pred = self.output_layer(x4).permute(0, 2, 3, 1)
            outputs.append(next_pred)
            current_input = next_pred
            X = new_x
        
        # Combine predictions
        outputs = torch.cat(outputs, dim=1)  # [batch_size, output_window, num_nodes, output_dim]
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
        
        # Handle scaler properly to avoid zero loss
        if self._scaler is not None:
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        else:
            # If no scaler (or NoneScaler), just take the values directly
            y_true = y_true[..., :self.output_dim]
            y_predicted = y_predicted[..., :self.output_dim]
        
        # Use MSE loss instead of MAE to ensure training stability
        return loss.masked_mse_torch(y_predicted, y_true, 0)