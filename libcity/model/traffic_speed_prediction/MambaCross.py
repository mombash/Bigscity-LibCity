import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model import loss
from libcity.model.traffic_speed_prediction.MCSTMamba import MCSTMamba

class GatedCrossAttention(nn.Module):
    """Gated Cross Attention module for temporal-spatial feature interaction"""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Multi-head attention layers
        self.temp_to_spat_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.spat_to_temp_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanisms
        self.temp_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        self.spat_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        
        # Output projections
        self.temp_output_proj = nn.Linear(d_model, d_model)
        self.spat_output_proj = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.temp_norm = nn.LayerNorm(d_model)
        self.spat_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_temp, x_spat):
        """
        Args:
            x_temp: [batch, time, nodes, d_model] - temporal features
            x_spat: [batch, time, nodes, d_model] - spatial features
        Returns:
            x_combined: [batch, time, nodes, d_model] - fused features
        """
        batch_size, time_steps, num_nodes, d_model = x_temp.shape
        
        # Reshape for attention: [batch * time, nodes, d_model]
        x_temp_reshaped = x_temp.view(batch_size * time_steps, num_nodes, d_model)
        x_spat_reshaped = x_spat.view(batch_size * time_steps, num_nodes, d_model)
        
        # Cross attention: temporal attends to spatial
        temp_attended, _ = self.temp_to_spat_attention(
            query=x_temp_reshaped,
            key=x_spat_reshaped,
            value=x_spat_reshaped
        )
        
        # Cross attention: spatial attends to temporal
        spat_attended, _ = self.spat_to_temp_attention(
            query=x_spat_reshaped,
            key=x_temp_reshaped,
            value=x_temp_reshaped
        )
        
        # Apply gating mechanisms
        temp_gate = self.temp_gate(x_temp_reshaped)
        spat_gate = self.spat_gate(x_spat_reshaped)
        
        # Gated fusion
        temp_fused = temp_gate * temp_attended + (1 - temp_gate) * x_temp_reshaped
        spat_fused = spat_gate * spat_attended + (1 - spat_gate) * x_spat_reshaped
        
        # Apply layer normalization and output projection
        temp_fused = self.temp_norm(temp_fused)
        spat_fused = self.spat_norm(spat_fused)
        
        temp_fused = self.temp_output_proj(temp_fused)
        spat_fused = self.spat_output_proj(spat_fused)
        
        # Combine temporal and spatial features
        x_combined = temp_fused + spat_fused
        x_combined = self.dropout(x_combined)
        
        # Reshape back to original dimensions
        x_combined = x_combined.view(batch_size, time_steps, num_nodes, d_model)
        
        return x_combined

class MambaCross(MCSTMamba):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        
        # Cross attention configuration
        self.num_heads = config.get('num_heads', 8)
        self.attention_dropout = config.get('attention_dropout', 0.1)
        
        # Replace gated fusion with gated cross attention
        self.gated_cross_attention = GatedCrossAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.attention_dropout
        )

        self.temp_proj = nn.Linear(self.d_model, self.d_model)
        self.spat_proj = nn.Linear(self.d_model, self.d_model)
        
        # Log the model configuration
        self._logger.info("Building MambaCross model with gated cross attention mechanism")

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
        
        # Prepare inputs for gated cross attention
        x_temp = x_temporal.permute(1, 2, 0, 3)  # [batch, time, nodes, d_model]
        x_spat = x_spatial.permute(1, 0, 2, 3)   # [batch, time, nodes, d_model]

        x_temp = self.temp_proj(x_temp)
        x_spat = self.spat_proj(x_spat)
        
        # Apply gated cross attention
        x_combined = self.gated_cross_attention(x_temp, x_spat)
        
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