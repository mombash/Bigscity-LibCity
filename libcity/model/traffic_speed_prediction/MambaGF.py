import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model import loss
from libcity.model.traffic_speed_prediction.MCSTMamba import MCSTMamba

class MambaGF(MCSTMamba):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        
        # Add gate projection layer for gated fusion
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.gate_dropout = config.get('gate_dropout', 0.1)
        self.gate_dropout_layer = nn.Dropout(self.gate_dropout)

        self.temp_proj = nn.Linear(self.d_model, self.d_model)
        self.spat_proj = nn.Linear(self.d_model, self.d_model)
        
        # Log the model configuration
        self._logger.info("Building MambaGF model with gated fusion mechanism")

    def forward(self, batch):
        # Move input to device if needed
        x = batch['X'].to(self.device)  # [batch_size, input_window, num_nodes, feature_dim]
        batch_size = x.shape[0]
        
        # Feature extraction
        features = []
        
        # Process main features
        x_main = self.input_proj(x)  # [batch_size, input_window, num_nodes, input_embedding_dim]
        features.append(x_main)
        
        # Add time embeddings if needed
        if self.add_time_in_day:
            tod = torch.linspace(0, 0.99, self.input_window, device=self.device)
            tod = tod.reshape(1, -1, 1).expand(batch_size, -1, self.num_nodes)
            tod_indices = (tod * self.steps_per_day).long()
            tod_indices = torch.clamp(tod_indices, 0, self.steps_per_day - 1)
            tod_emb = self.tod_embedding(tod_indices)
            features.append(tod_emb)
            
        if self.add_day_in_week:
            dow = torch.arange(0, self.input_window, device=self.device) % 7
            dow = dow.reshape(1, -1, 1).expand(batch_size, -1, self.num_nodes)
            dow_indices = dow.long()
            dow_indices = torch.clamp(dow_indices, 0, 6)
            dow_emb = self.dow_embedding(dow_indices)
            features.append(dow_emb)
        
        # Add spatial embeddings
        spatial_emb = self.spatial_embedding.unsqueeze(0).unsqueeze(0)
        spatial_emb = spatial_emb.expand(batch_size, self.input_window, -1, -1)
        features.append(spatial_emb)
        
        # Add adaptive embeddings if enabled
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.unsqueeze(0)
            adp_emb = adp_emb.expand(batch_size, -1, -1, -1)
            features.append(adp_emb)
        
        # Concatenate all features
        x = torch.cat(features, dim=-1)
        
        # Project to Mamba dimension
        x = self.mamba_input_proj(x)
        
        # Temporal processing
        x_temporal = x.permute(2, 0, 1, 3)
        x_temporal = x_temporal.reshape(self.num_nodes, batch_size * self.input_window, -1)
        x_temporal = self.temporal_block(x_temporal)
        x_temporal = x_temporal.reshape(self.num_nodes, batch_size, self.input_window, self.d_model)
        
        # Spatial processing
        is_large_dataset = self.num_nodes > 300
        
        if is_large_dataset and batch_size > 1:
            x_spatial = torch.zeros(self.input_window, batch_size, self.num_nodes, self.d_model, 
                                  device=self.device)
            
            for t in range(self.input_window):
                nodes_seq = x[:, t, :, :]
                effective_batch_size = 1
                
                all_results = []
                for b_idx in range(0, batch_size, effective_batch_size):
                    end_idx = min(b_idx + effective_batch_size, batch_size)
                    batch_slice = nodes_seq[b_idx:end_idx]
                    spatial_hidden = self.spatial_block(batch_slice)
                    all_results.append(spatial_hidden)
                
                x_spatial[t] = torch.cat(all_results, dim=0)
        else:
            x_spatial = x.permute(1, 0, 2, 3)
            x_spatial = x_spatial.reshape(self.input_window, batch_size * self.num_nodes, -1)
            x_spatial = self.spatial_block(x_spatial)
            x_spatial = x_spatial.reshape(self.input_window, batch_size, self.num_nodes, self.d_model)
        
        # Gated fusion mechanism
        x_temp = x_temporal.permute(1, 2, 0, 3)  # [batch, time, nodes, d_model]
        x_spat = x_spatial.permute(1, 0, 2, 3)   # [batch, time, nodes, d_model]

        x_temp = self.temp_proj(x_temp)
        x_spat = self.spat_proj(x_spat)
        
        # Concatenate temporal and spatial features
        fusion_input = torch.cat([x_temp, x_spat], dim=-1)  # [batch, time, nodes, 2*d_model]
        
        # Compute the gate without dropout before sigmoid
        gate = torch.sigmoid(self.gate_proj(fusion_input))  # [batch, time, nodes, d_model]
        
        # Fuse the outputs
        x_combined = gate * x_temp + (1 - gate) * x_spat
        
        # Apply dropout after fusion step
        x_combined = self.gate_dropout_layer(x_combined)
        
        # Final processing and output projection
        x_out = self.final_layer_norm(x_combined)
        x_out = self.output_proj(x_out)
        
        return x_out[:, -self.output_window:]  # Return last output_window steps

    def calculate_loss(self, batch):
        y_true = batch['y'].to(self.device)
        y_predicted = self.predict(batch)
        
        # Apply inverse normalization
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        
        # Calculate masked MAE loss
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch) 