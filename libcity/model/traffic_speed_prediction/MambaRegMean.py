import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from logging import getLogger
from libcity.model import loss
from libcity.model.traffic_speed_prediction.MCSTMamba import MCSTMamba
from tqdm import tqdm

class MambaRegMean(MCSTMamba):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        
        # RegMean configuration
        self.use_regmean = config.get('use_regmean', True)
        self.regmean_steps = config.get('regmean_steps', 1000)
        self.regmean_floor = config.get('regmean_floor', 1e-6)
        
        # Store gram matrices for each layer/module (matching regmean_script.py approach)
        self.grams = {}  # gram matrices for each layer/module
        self.gram_counts = {}  # number of samples used for computing gram
        
        # Hook handles for cleanup
        self.hook_handles = []
        
        # Log the model configuration
        self._logger.info("Building MambaRegMean model with RegMean fusion mechanism")
        
        # Setup hooks for gram matrix computation
        if self.use_regmean:
            self._setup_gram_hooks()

    def _setup_gram_hooks(self):
        """Setup hooks to capture activations for gram matrix computation."""
        def get_gram(name):
            def hook(module, input, output):
                if not self.training:
                    return
                    
                x = input[0].detach()  # [batch_size, seq_len, num_nodes, d_model] or similar
                x = x.view(-1, x.size(-1))  # Flatten to [total_elements, d_model]
                
                # Compute gram matrix: X^T * X
                xtx = torch.matmul(x.transpose(0, 1), x)
                
                # Update gram matrices using running average
                if name not in self.grams:
                    self.grams[name] = xtx / x.size(0)
                    self.gram_counts[name] = x.size(0)
                else:
                    self.grams[name] = (self.grams[name] * self.gram_counts[name] + xtx) / (self.gram_counts[name] + x.size(0))
                    self.gram_counts[name] += x.size(0)
            return hook

        # Register hooks on all linear layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_gram(name))
                self.hook_handles.append(handle)

    def compute_gram_matrix(self, x, feature_type):
        """
        Compute Gram matrix for input features.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes, d_model]
            feature_type: 'temporal' or 'spatial'
        """
        batch_size, seq_len, num_nodes, d_model = x.shape
        
        # Ensure tensor is contiguous and reshape to [total_elements, d_model]
        x_reshaped = x.contiguous().reshape(-1, d_model)
        
        # Compute Gram matrix: X^T * X
        gram_matrix = torch.matmul(x_reshaped.transpose(0, 1), x_reshaped)
        
        # Normalize by number of elements
        gram_matrix = gram_matrix / x_reshaped.size(0)
        
        return gram_matrix

    def update_gram_matrices(self, x_temporal, x_spatial):
        """
        Update stored Gram matrices with new batch data using running average approach.
        This matches the approach used in regmean_script.py.
        
        Args:
            x_temporal: Temporal features [batch_size, seq_len, num_nodes, d_model]
            x_spatial: Spatial features [batch_size, seq_len, num_nodes, d_model]
        """
        if not self.training or not self.use_regmean:
            return
            
        # Compute Gram matrices for current batch
        temporal_gram = self.compute_gram_matrix(x_temporal, 'temporal')
        spatial_gram = self.compute_gram_matrix(x_spatial, 'spatial')
        
        # Detach from computation graph to avoid backward issues
        temporal_gram = temporal_gram.detach()
        spatial_gram = spatial_gram.detach()
        
        # Update stored Gram matrices using running average (matching regmean_script.py)
        if 'temporal' not in self.grams:
            self.grams['temporal'] = temporal_gram
            self.grams['spatial'] = spatial_gram
            self.gram_counts['temporal'] = x_temporal.reshape(-1, x_temporal.size(-1)).size(0)
            self.gram_counts['spatial'] = x_spatial.reshape(-1, x_spatial.size(-1)).size(0)
        else:
            # Running average: (old_gram * old_count + new_gram * new_count) / (old_count + new_count)
            temp_count = x_temporal.reshape(-1, x_temporal.size(-1)).size(0)
            spat_count = x_spatial.reshape(-1, x_spatial.size(-1)).size(0)
            
            self.grams['temporal'] = (self.grams['temporal'] * self.gram_counts['temporal'] + 
                                    temporal_gram * temp_count) / (self.gram_counts['temporal'] + temp_count)
            self.grams['spatial'] = (self.grams['spatial'] * self.gram_counts['spatial'] + 
                                   spatial_gram * spat_count) / (self.gram_counts['spatial'] + spat_count)
            
            self.gram_counts['temporal'] += temp_count
            self.gram_counts['spatial'] += spat_count

    def regmean_merge_features(self, x_temporal, x_spatial):
        """
        Merge temporal and spatial features using RegMean.
        
        Args:
            x_temporal: Temporal features [batch_size, seq_len, num_nodes, d_model]
            x_spatial: Spatial features [batch_size, seq_len, num_nodes, d_model]
            
        Returns:
            Merged features [batch_size, seq_len, num_nodes, d_model]
        """
        if not self.use_regmean or 'temporal' not in self.grams:
            # Fallback to simple averaging if RegMean is disabled or no Gram matrices
            return (x_temporal + x_spatial) / 2.0
        
        # Get stored Gram matrices
        temporal_gram = self.grams['temporal']
        spatial_gram = self.grams['spatial']
        
        # Ensure Gram matrices are on the same device as input tensors
        device = x_temporal.device
        temporal_gram = temporal_gram.to(device)
        spatial_gram = spatial_gram.to(device)
        
        # Add regularization to ensure invertibility
        reg_factor = self.regmean_floor
        temporal_gram_reg = temporal_gram + reg_factor * torch.eye(temporal_gram.size(0), device=device)
        spatial_gram_reg = spatial_gram + reg_factor * torch.eye(spatial_gram.size(0), device=device)
        
        # Compute inverse of combined Gram matrix
        combined_gram = temporal_gram_reg + spatial_gram_reg
        try:
            combined_gram_inv = torch.inverse(combined_gram)
        except:
            # Fallback to pseudo-inverse if matrix is singular
            combined_gram_inv = torch.pinverse(combined_gram)
        
        # Reshape features for matrix operations
        batch_size, seq_len, num_nodes, d_model = x_temporal.shape
        x_temp_reshaped = x_temporal.contiguous().reshape(-1, d_model)  # [batch*seq*nodes, d_model]
        x_spat_reshaped = x_spatial.contiguous().reshape(-1, d_model)   # [batch*seq*nodes, d_model]
        
        # Apply RegMean formula: (G1 + G2)^(-1) * (G1 * x1 + G2 * x2)
        # where G1, G2 are Gram matrices and x1, x2 are features
        
        # Compute weighted features
        weighted_temp = torch.matmul(temporal_gram, x_temp_reshaped.transpose(0, 1))  # [d_model, batch*seq*nodes]
        weighted_spat = torch.matmul(spatial_gram, x_spat_reshaped.transpose(0, 1))   # [d_model, batch*seq*nodes]
        
        # Combine weighted features
        combined_weighted = weighted_temp + weighted_spat  # [d_model, batch*seq*nodes]
        
        # Apply inverse
        merged_features = torch.matmul(combined_gram_inv, combined_weighted)  # [d_model, batch*seq*nodes]
        
        # Reshape back to original shape
        merged_features = merged_features.transpose(0, 1).reshape(batch_size, seq_len, num_nodes, d_model)
        
        return merged_features

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
        
        # Prepare features for RegMean fusion
        x_temp = x_temporal.permute(1, 2, 0, 3)  # [batch, time, nodes, d_model]
        x_spat = x_spatial.permute(1, 0, 2, 3)   # [batch, time, nodes, d_model]
        
        # Update Gram matrices during training (hooks handle this automatically)
        # The hooks will capture activations from all linear layers during forward pass
        
        # RegMean fusion mechanism
        x_combined = self.regmean_merge_features(x_temp, x_spat)
        
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

    def __del__(self):
        """Cleanup hooks when model is destroyed."""
        for handle in self.hook_handles:
            handle.remove() 