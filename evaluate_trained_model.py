#!/usr/bin/env python
"""
This script evaluates a previously trained traffic state prediction model on its test dataset.
It can be used to evaluate any model in the LibCity framework.
"""

'''
python evaluate_trained_model.py \
  --model Mamba \
  --dataset PEMSD8 \
  --model_dir Mamba_PEMSD8_20250408_141451 \
  --epoch 95 \
  --evaluate_channels_separately
'''

import os
import argparse
import torch
import json
import glob
import numpy as np
import pandas as pd
import time
import shutil
import matplotlib.pyplot as plt
from libcity.utils import get_executor, get_model, get_logger, str2bool, ensure_dir, get_evaluator
from libcity.config import ConfigParser
from libcity.data import get_dataset


def patch_model_for_checkpoint(model, model_name, checkpoint):
    """
    Patch the model's architecture parameters to match the checkpoint.
    This is particularly important for models like MCSTMamba where certain
    parameters are hardcoded.
    
    Args:
        model: The model to patch
        model_name: Name of the model class
        checkpoint: The loaded checkpoint
    """
    if model_name != 'MCSTMamba':
        return
    
    # For MCSTMamba, we need to update parameters
    try:
        # Try to extract key dimensions from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Look for A_log to get d_state
        a_log_keys = [k for k in state_dict.keys() if k.endswith('.mamba.A_log')]
        if a_log_keys:
            key = a_log_keys[0]
            d_state = state_dict[key].shape[1]
            if hasattr(model, 'd_state'):
                print(f"Patching model d_state from {model.d_state} to {d_state}")
                model.d_state = d_state
        
        # Look for x_proj.weight to calculate expanded dt_size
        x_proj_keys = [k for k in state_dict.keys() if k.endswith('.mamba.x_proj.weight')]
        if x_proj_keys:
            key = x_proj_keys[0]
            x_proj_first_dim = state_dict[key].shape[0]
            # This value is often related to dt_size
            if hasattr(model, 'dt_size'):
                # Common formula: dt_size = x_proj_first_dim - 6
                dt_size = x_proj_first_dim - 6
                print(f"Patching model dt_size from {model.dt_size if hasattr(model, 'dt_size') else 'unknown'} to {dt_size}")
                model.dt_size = dt_size
                
    except Exception as e:
        print(f"Error patching model parameters: {e}")


def adapt_checkpoint_to_model(checkpoint, model_name):
    """
    Adjust the checkpoint to match the current model architecture,
    or vice versa. This is especially useful for Mamba models where
    shape mismatches can occur due to hyperparameter changes.
    
    Args:
        checkpoint: The loaded checkpoint
        model_name: Name of the model
        
    Returns:
        Modified checkpoint that should load correctly
    """
    if model_name != 'MCSTMamba':
        return checkpoint
    
    # Make a copy to avoid modifying the original
    checkpoint_copy = {}
    for key in checkpoint:
        if key == 'model_state_dict':
            # Create a deep copy of the state dict
            checkpoint_copy[key] = {}
            for param_name, param in checkpoint[key].items():
                checkpoint_copy[key][param_name] = param.clone() if isinstance(param, torch.Tensor) else param
        else:
            checkpoint_copy[key] = checkpoint[key]
    
    # Check for common MCSTMamba issues
    if 'model_state_dict' in checkpoint_copy:
        state_dict = checkpoint_copy['model_state_dict']
        modified = False
        
        # Issue 1: Check A_log parameters - these define the state dimension
        for key in list(state_dict.keys()):
            if key.endswith('.mamba.A_log'):
                # Actual resizing will be handled in load_model_safely
                print(f"Found potential A_log mismatch in {key}: {state_dict[key].shape}")
                modified = True
        
        # Issue 2: Check x_proj.weight - these determine dt_size + d_state
        for key in list(state_dict.keys()):
            if key.endswith('.mamba.x_proj.weight'):
                # Actual resizing will be handled in load_model_safely
                print(f"Found potential x_proj.weight mismatch in {key}: {state_dict[key].shape}")
                modified = True
    
    if modified:
        print("Checkpoint has been identified as requiring adaptation during loading")
    return checkpoint_copy


def load_model_safely(executor, model_path, epoch=None, model_name=None):
    """
    Safely load a model by adapting the checkpoint if necessary.
    
    Args:
        executor: The model executor
        model_path: Path to the model checkpoint
        epoch: Specific epoch number if loading with epoch
        model_name: Name of the model class
    """
    try:
        # Standard loading approach
        if epoch is not None:
            executor.load_model_with_epoch(epoch)
        else:
            executor.load_model(model_path)
            
    except RuntimeError as e:
        # Check if it's a size mismatch error
        if "size mismatch" in str(e):
            print("Detected size mismatch. Attempting to adapt checkpoint...")
            
            # For MCSTMamba specifically
            if model_name == 'MCSTMamba' or (hasattr(executor.model, '__class__') and 
                                             executor.model.__class__.__name__ == 'MCSTMamba'):
                # Load the checkpoint
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # First try to patch the model's parameters to match the checkpoint
                patch_model_for_checkpoint(executor.model, model_name, checkpoint)
                
                # Adapt the checkpoint to the model
                adapted_checkpoint = adapt_checkpoint_to_model(checkpoint, model_name)
                
                # Get model state dict and original architecture
                state_dict = adapted_checkpoint['model_state_dict']
                current_state_dict = executor.model.state_dict()
                
                # Create new state dict compatible with current model
                new_state_dict = {}
                
                # Find compatible layers and load them
                for name, param in state_dict.items():
                    if name in current_state_dict:
                        # If shapes match, copy directly
                        if current_state_dict[name].shape == param.shape:
                            new_state_dict[name] = param
                        else:
                            # For A_log, resize if necessary
                            if name.endswith('.mamba.A_log'):
                                saved_shape = param.shape
                                current_shape = current_state_dict[name].shape
                                
                                # Shape is typically [d_inner, d_state]
                                if saved_shape[0] == current_shape[0]:  # If first dim matches
                                    # Either pad or trim to match
                                    if saved_shape[1] < current_shape[1]:
                                        # Pad with zeros
                                        padded = torch.zeros(current_shape, device=param.device, dtype=param.dtype)
                                        padded[:, :saved_shape[1]] = param
                                        new_state_dict[name] = padded
                                        print(f"Padded {name} from {saved_shape} to {current_shape}")
                                    else:
                                        # Trim
                                        new_state_dict[name] = param[:, :current_shape[1]]
                                        print(f"Trimmed {name} from {saved_shape} to {current_shape}")
                                    
                            # For x_proj.weight
                            elif name.endswith('.mamba.x_proj.weight'):
                                saved_shape = param.shape
                                current_shape = current_state_dict[name].shape
                                
                                if saved_shape[1] == current_shape[1]:  # If second dim matches
                                    if saved_shape[0] < current_shape[0]:
                                        # Pad with zeros
                                        padded = torch.zeros(current_shape, device=param.device, dtype=param.dtype)
                                        padded[:saved_shape[0], :] = param
                                        new_state_dict[name] = padded
                                        print(f"Padded {name} from {saved_shape} to {current_shape}")
                                    else:
                                        # Trim
                                        new_state_dict[name] = param[:current_shape[0], :]
                                        print(f"Trimmed {name} from {saved_shape} to {current_shape}")
                            else:
                                print(f"Skipping incompatible parameter: {name}, "
                                     f"saved shape: {param.shape}, current shape: {current_state_dict[name].shape}")
                    else:
                        print(f"Parameter {name} not found in current model")
                
                # Load compatible parameters
                if new_state_dict:
                    missing = set(current_state_dict.keys()) - set(new_state_dict.keys())
                    if missing:
                        print(f"Warning: {len(missing)} parameters not loaded. Model may not behave as expected.")
                        
                    executor.model.load_state_dict(new_state_dict, strict=False)
                    print("Loaded partial state dict with compatible parameters")
                else:
                    print("No compatible parameters found. Using original model initialization.")
            else:
                # For other models, re-raise the exception
                raise
        else:
            # If it's not a size mismatch, re-raise the exception
            raise


def visualize_single_channel(y_true, y_pred, save_dir, filename, title, batch_idx):
    """
    Create a visualization for a single channel of data.
    
    Args:
        y_true: Ground truth values (tensor or numpy)
        y_pred: Predicted values (tensor or numpy)
        save_dir: Directory to save the visualization
        filename: Filename for the visualization
        title: Title for the plot
        batch_idx: Batch index for sample selection
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    if y_true.shape[0] == 0 or y_true.shape[2] == 0:
        print(f"Skipping visualization for {title}, batch {batch_idx}: Empty data")
        return
    
    # Select a random sample from the batch
    sample_idx = np.random.randint(0, y_true.shape[0])
    
    # Select a few random nodes (at most 4)
    num_nodes = min(4, y_true.shape[2])
    node_indices = np.random.choice(y_true.shape[2], size=num_nodes, replace=False)
    
    # Create the figure
    plt.figure(figsize=(15, 10))
    
    # Plot each node
    for i, node_idx in enumerate(node_indices):
        plt.subplot(2, 2, i+1)
        
        # Extract data for this node
        true_data = y_true[sample_idx, :, node_idx, 0]  # Channel is already selected
        pred_data = y_pred[sample_idx, :, node_idx, 0]
        
        # Create plot
        plt.plot(true_data, 'b-', label='Ground Truth', marker='o')
        plt.plot(pred_data, 'r--', label='Prediction', marker='x')
        plt.title(f"{title} - Node {node_idx}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
    
    plt.suptitle(f"{title} - Sample {sample_idx}, Batch {batch_idx}")
    plt.tight_layout()
    
    # Save the figure
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()


def evaluate_single_channel(executor, test_dataloader, channel_idx, channel_name, save_dir, logger):
    """
    Evaluate a model on a single channel of the dataset.
    
    Args:
        executor: Model executor
        test_dataloader: Test data
        channel_idx: Channel index to evaluate (0-based)
        channel_name: Name for this channel evaluation
        save_dir: Directory to save results
        logger: Logger instance
    
    Returns:
        Evaluation results for this channel
    """
    logger.info(f"Evaluating {channel_name} (index {channel_idx})...")
    
    # Make sure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect predictions and ground truth
    y_truths = []
    y_preds = []
    
    with torch.no_grad():
        executor.model.eval()
        
        for batch_idx, batch in enumerate(test_dataloader):
            batch.to_tensor(executor.device)
            
            # Get predictions
            output = executor.model.predict(batch)
            
            # Get only the specific channel
            y_true = executor._scaler.inverse_transform(batch['y'][..., channel_idx:channel_idx+1])
            y_pred = executor._scaler.inverse_transform(output[..., channel_idx:channel_idx+1])
            
            # Add to lists
            y_truths.append(y_true.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
            
    # Concatenate results
    y_truths_concat = np.concatenate(y_truths, axis=0)
    y_preds_concat = np.concatenate(y_preds, axis=0)
    
    # Save predictions
    outputs = {'prediction': y_preds_concat, 'truth': y_truths_concat}
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    filename = f"{timestamp}_{executor.config['model']}_{executor.config['dataset']}_{channel_name}_predictions.npz"
    filepath = os.path.join(save_dir, filename)
    
    try:
        np.savez_compressed(filepath, **outputs)
        logger.info(f"Saved {channel_name} predictions to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save {channel_name} predictions: {e}")
    
    # Calculate metrics
    try:
        # Create a new evaluator to avoid affecting the main one
        evaluator = get_evaluator(executor.config)
        
        # Collect data
        evaluator.clear()
        evaluator.collect({'y_true': torch.tensor(y_truths_concat), 'y_pred': torch.tensor(y_preds_concat)})
        
        # Save results
        channel_results = evaluator.save_result(save_dir, f"{channel_name}_metrics")
        logger.info(f"Saved {channel_name} metrics to {save_dir}")
        
        return channel_results
    except Exception as e:
        logger.error(f"Failed to evaluate {channel_name}: {e}")
        return None


def evaluate_model(task='traffic_state_pred', model_name=None, dataset_name=None, 
                  model_dir=None, epoch=None, config_file=None, other_args=None,
                  evaluate_channels_separately=False):
    """
    Evaluate a previously trained model on the test dataset.
    
    Args:
        task (str): Task name, default to traffic_state_pred
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
        model_dir (str): Directory containing the model cache files (exp_id directory)
        epoch (int): Specific epoch model to load, if not provided uses the general model file
        config_file (str): Config file to override settings
        other_args (dict): Other arguments to pass to the config parser
        evaluate_channels_separately (bool): If True, evaluate each output channel separately
    """
    if other_args is None:
        other_args = {}
    
    # Get exp_id from model_dir if not explicitly set
    if model_dir and '/' in model_dir:
        exp_id = os.path.basename(os.path.normpath(model_dir))
    else:
        exp_id = model_dir
    
    # Create a unique ID for this evaluation run
    eval_timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_run_id = f"eval_{eval_timestamp}"
    
    # Adjust model cache paths based on model_dir
    if model_dir:
        model_cache_dir = os.path.join('./libcity/cache', exp_id, 'model_cache')
        # Create a new folder for this evaluation run
        eval_dir = os.path.join('./libcity/cache', exp_id, 'evaluations', eval_run_id)
        os.makedirs(eval_dir, exist_ok=True)
        print(f"Created evaluation directory: {eval_dir}")
        
        # Create visualizations subfolder
        vis_dir = os.path.join(eval_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Created visualizations directory: {vis_dir}")
    else:
        raise ValueError("model_dir is required to locate the model files")
    
    # Determine the model path based on inputs
    if epoch is not None:
        # Specific epoch model
        model_path = os.path.join(model_cache_dir, f"{model_name}_{dataset_name}_epoch{epoch}.tar")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    else:
        # Use the general model file
        model_path = os.path.join(model_cache_dir, f"{model_name}_{dataset_name}.m")
        if not os.path.exists(model_path):
            # If the general file doesn't exist, try to find the highest epoch model
            pattern = os.path.join(model_cache_dir, f"{model_name}_{dataset_name}_epoch*.tar")
            model_files = glob.glob(pattern)
            if not model_files:
                raise FileNotFoundError(f"No model files found matching pattern: {pattern}")
            
            # Extract epoch numbers and find the highest
            epochs = [int(f.split('_epoch')[-1].split('.')[0]) for f in model_files]
            highest_epoch = max(epochs)
            model_path = os.path.join(model_cache_dir, f"{model_name}_{dataset_name}_epoch{highest_epoch}.tar")
            print(f"Using highest epoch model: {model_path}")
    
    # Before initializing the model, try to extract the configuration from the saved model
    print(f"Loading model configuration from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Check if configuration is stored in the checkpoint
    if 'config' in checkpoint:
        # Use the configuration from the checkpoint
        model_config = checkpoint['config']
        print(f"Using model configuration from checkpoint")
        
        # Add the extracted config to other_args to override the defaults
        for key, value in model_config.items():
            other_args[key] = value
        
    elif model_name == 'MCSTMamba':
        # MCSTMamba specific: Try to extract key architecture parameters from model state dict
        try:
            state_dict = checkpoint['model_state_dict']
            
            # Extract d_state parameter from A_log shape
            a_log_key = 'temporal_block.mamba.A_log'
            if a_log_key in state_dict:
                d_state = state_dict[a_log_key].shape[1]
                other_args['d_state'] = d_state
                print(f"Extracted d_state={d_state} from model weights")
            
            # Extract parameters from other tensor shapes if needed
            # For example, x_proj.weight shape for Mamba-specific parameters
            x_proj_key = 'temporal_block.mamba.x_proj.weight'
            if x_proj_key in state_dict:
                x_proj_shape = state_dict[x_proj_key].shape[0]
                # In Mamba, this is dt_size + d_state, or specific to some implementations
                # Set the exact value to ensure proper loading
                other_args['dt_size'] = x_proj_shape - 6
                print(f"Extracted x_proj first dimension={x_proj_shape} from model weights")
                print(f"Setting dt_size={x_proj_shape-6} to match the saved model")
                
                # If MCSTMamba, we need explicit handling since the model has hardcoded values
                # that might not match what's in the checkpoint
                if model_name == 'MCSTMamba':
                    # Add a marker to be checked in the model's __init__ method
                    # Since we can't modify the model code, we'll find another approach
                    # This may require updating the checkpoint itself
                    print(f"For MCSTMamba, explicitly specifying exact x_proj dimension: {x_proj_shape}")
                    other_args['x_proj_first_dim'] = x_proj_shape
            
        except Exception as e:
            print(f"Warning: Could not fully extract model parameters from state dict: {e}")
            print("Will try to load with default parameters, but this may cause compatibility issues")
    
    # Add the eval_dir to other_args to override the default evaluate_dir
    other_args['evaluate_res_dir'] = eval_dir
    # Add the visualizations directory
    other_args['visualization_dir'] = vis_dir
    # Set visualization options
    other_args['save_mode'] = ['csv', 'png']
    other_args['save_plots'] = True
    
    # Construct the config with potentially updated parameters
    config = ConfigParser(task, model_name, dataset_name, 
                         config_file, saved_model=False, train=False, other_args=other_args)
    config['exp_id'] = exp_id
    
    # Setup logger
    logger = get_logger(config)
    logger.info(f"Evaluating model: {model_name} on dataset: {dataset_name}")
    logger.info(f"Using model from: {model_path}")
    logger.info(f"Evaluation results will be saved to: {eval_dir}")
    logger.info(f"Visualizations will be saved to: {vis_dir} (see other_args['visualization_dir'])")
    
    # Write evaluation parameters to a file in the eval directory
    eval_params = {
        'task': task,
        'model': model_name,
        'dataset': dataset_name,
        'model_dir': model_dir,
        'epoch': epoch,
        'eval_timestamp': eval_timestamp,
        'config_overrides': other_args
    }
    
    # Save evaluation parameters as JSON
    with open(os.path.join(eval_dir, 'eval_params.json'), 'w') as f:
        json.dump(eval_params, f, indent=2)
    
    # Load the dataset
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    
    # Initialize the model with the extracted configuration
    model = get_model(config, data_feature)
    
    # Ensure output_dim is set in the config
    if 'output_dim' not in config:
        if hasattr(model, 'output_dim'):
            config['output_dim'] = model.output_dim
            logger.info(f"Setting output_dim from model: {model.output_dim}")
        else:
            config['output_dim'] = data_feature.get('output_dim', 1)
            logger.info(f"Setting output_dim from data_feature: {config['output_dim']}")
    
    # Initialize the executor
    executor = get_executor(config, model, data_feature)
    
    # IMPORTANT: Update the executor's visualization directory to our custom path
    # This works because the executor already has 'visualization_dir' as an instance variable
    original_vis_dir = executor.visualization_dir if hasattr(executor, 'visualization_dir') else "Not set"
    executor.visualization_dir = vis_dir
    logger.info(f"Updated visualization directory from {original_vis_dir} to {vis_dir}")
    
    # Verify executor attributes
    logger.info(f"Executor type: {type(executor).__name__}")
    logger.info(f"Executor has visualization_dir: {hasattr(executor, 'visualization_dir')}")
    if hasattr(executor, 'visualization_dir'):
        logger.info(f"Final executor.visualization_dir: {executor.visualization_dir}")
        
    # Verify directory exists 
    if os.path.exists(vis_dir):
        logger.info(f"Visualization directory exists: {vis_dir}")
    else:
        logger.warning(f"Creating visualization directory: {vis_dir}")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Load the model
    try:
        # Load model with our safe loading function
        load_model_safely(executor, model_path, epoch, model_name)
        
        # Evaluate the model
        logger.info("Starting evaluation...")
        
        if evaluate_channels_separately:
            logger.info("Evaluating each channel separately...")
            
            # Get the number of output channels
            num_channels = data_feature.get('output_dim', 1)
            logger.info(f"Dataset has {num_channels} output channels")
            
            # Create a subdirectory for per-channel results
            per_channel_dir = os.path.join(eval_dir, 'per_channel')
            os.makedirs(per_channel_dir, exist_ok=True)
            
            all_results = {}
            
            # For each channel
            for channel in range(num_channels):
                logger.info(f"Evaluating channel {channel + 1}/{num_channels}")
                
                # Create a new evaluator for this channel
                per_channel_results = evaluate_single_channel(
                    executor, test_data, channel, 
                    f"channel_{channel}", per_channel_dir, logger
                )
                
                # Store results
                all_results[f"channel_{channel}"] = per_channel_results
            
            # Also run the standard evaluation on all channels
            logger.info("Evaluating all channels together (standard)...")
            standard_results = evaluate_with_batch_timing(executor, test_data, logger)
            all_results["all_channels"] = standard_results
            
            # Save combined results
            combined_results_file = os.path.join(eval_dir, 'combined_results.json')
            with open(combined_results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Saved combined results to {combined_results_file}")
            
            # Use standard results as the main results
            results = standard_results
        else:
            # Standard evaluation (all channels together) with batch timing
            results = evaluate_with_batch_timing(executor, test_data, logger)
        
        # End timing the evaluation process
        eval_end_time = time.time()
        eval_duration = eval_end_time - eval_start_time
        logger.info(f"Evaluation completed in {eval_duration:.2f} seconds")
        
        # Log the results
        logger.info("Evaluation complete!")
        logger.info(f"Results: {results}")
        
        # Save the results in CSV format
        try:
            # For separate channel evaluation
            if evaluate_channels_separately:
                # Convert the combined results dictionary to a DataFrame in "long format"
                rows = []
                
                # Process all channels and the combined results
                for channel_key, channel_results in all_results.items():
                    if channel_results is None:
                        continue
                        
                    # Process this channel's results
                    for metric, value in channel_results.items():
                        if isinstance(value, (list, np.ndarray)):
                            # Create a row for each value in the array
                            for i, val in enumerate(value):
                                rows.append({
                                    'metric': metric,
                                    'channel': channel_key,
                                    'horizon': i+1,  # 1-indexed for horizons
                                    'value': float(val),
                                    'model': model_name,
                                    'dataset': dataset_name,
                                    'epoch': epoch,
                                    'timestamp': eval_timestamp
                                })
                        else:
                            # Single value
                            rows.append({
                                'metric': metric,
                                'channel': channel_key,
                                'horizon': 'all',  # Aggregate metric
                                'value': float(value) if isinstance(value, (int, float, np.number)) else value,
                                'model': model_name,
                                'dataset': dataset_name,
                                'epoch': epoch,
                                'timestamp': eval_timestamp
                            })
                
                # Create DataFrame with one row per individual value
                df = pd.DataFrame(rows)
                
                # Save to CSV
                csv_file = os.path.join(eval_dir, 'evaluation_results_all_channels.csv')
                df.to_csv(csv_file, index=False)
                logger.info(f"Saved all channel results to CSV: {csv_file}")
                
                # Save individual CSVs for each channel
                for channel_key in all_results.keys():
                    channel_df = df[df['channel'] == channel_key]
                    if not channel_df.empty:
                        channel_csv = os.path.join(eval_dir, f'evaluation_results_{channel_key}.csv')
                        channel_df.to_csv(channel_csv, index=False)
                        logger.info(f"Saved {channel_key} results to: {channel_csv}")
            
            # Standard CSV conversion for the main results
            # Convert the results dictionary to a DataFrame in "long format"
            # (each metric on its own row instead of wide format with all metrics in one row)
            rows = []
            for key, value in results.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        metric_name = f"{key}_{subkey}"
                        # Check if the value is an array/list
                        if isinstance(subvalue, (list, np.ndarray)):
                            # Create a row for each value in the array
                            for i, val in enumerate(subvalue):
                                rows.append({
                                    'metric': metric_name,
                                    'horizon': i+1,  # 1-indexed for horizons
                                    'value': float(val),
                                    'model': model_name,
                                    'dataset': dataset_name,
                                    'epoch': epoch,
                                    'timestamp': eval_timestamp
                                })
                        else:
                            # Single value
                            rows.append({
                                'metric': metric_name,
                                'horizon': 'all',  # Aggregate metric
                                'value': float(subvalue) if isinstance(subvalue, (int, float, np.number)) else subvalue,
                                'model': model_name,
                                'dataset': dataset_name,
                                'epoch': epoch,
                                'timestamp': eval_timestamp
                            })
                else:
                    # Check if the value is an array/list
                    if isinstance(value, (list, np.ndarray)):
                        # Create a row for each value in the array
                        for i, val in enumerate(value):
                            rows.append({
                                'metric': key,
                                'horizon': i+1,  # 1-indexed for horizons
                                'value': float(val),
                                'model': model_name,
                                'dataset': dataset_name,
                                'epoch': epoch,
                                'timestamp': eval_timestamp
                            })
                    else:
                        # Single value
                        rows.append({
                            'metric': key,
                            'horizon': 'all',  # Aggregate metric
                            'value': float(value) if isinstance(value, (int, float, np.number)) else value,
                            'model': model_name,
                            'dataset': dataset_name,
                            'epoch': epoch,
                            'timestamp': eval_timestamp
                        })
            
            # Create DataFrame with one row per individual value
            df = pd.DataFrame(rows)
            
            # Save to CSV
            csv_file = os.path.join(eval_dir, 'evaluation_results.csv')
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved results to CSV: {csv_file}")
            
            # Log how many rows were saved
            logger.info(f"CSV contains {len(df)} rows with individual metric values")
            
            # Also save as JSON for backward compatibility
            json_file = os.path.join(eval_dir, 'evaluation_results.json')
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")
            # Fall back to JSON
            json_file = os.path.join(eval_dir, 'evaluation_results.json')
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results to JSON: {json_file}")
        
        # Move visualization files to the visualization directory
        try:
            # Look for visualization files in the evaluation directory and evaluate_cache
            for search_dir in [eval_dir, os.path.join('./libcity/cache', exp_id, 'evaluate_cache')]:
                if os.path.exists(search_dir):
                    for file in os.listdir(search_dir):
                        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.pdf') or file.endswith('.svg'):
                            src_path = os.path.join(search_dir, file)
                            dst_path = os.path.join(vis_dir, file)
                            if os.path.exists(src_path):
                                # Use copy instead of move to avoid permission issues
                                shutil.copy2(src_path, dst_path)
                                logger.info(f"Copied visualization {file} to {vis_dir}")
                                # Try to remove the original file after copying
                                try:
                                    if search_dir != vis_dir:  # Don't remove if we're already in the vis dir
                                        os.remove(src_path)
                                        logger.info(f"Removed original visualization after copy: {src_path}")
                                except Exception as e:
                                    logger.warning(f"Could not remove original visualization: {e}")
            
            # Check if the visualization directory now has files
            vis_files = [f for f in os.listdir(vis_dir) if os.path.isfile(os.path.join(vis_dir, f))]
            logger.info(f"Visualization directory now contains {len(vis_files)} files")
            if len(vis_files) == 0:
                logger.warning("No visualization files were found or moved!")
                
        except Exception as e:
            logger.error(f"Error handling visualization files: {e}")
        
        return results
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='traffic_state_pred', 
                        help='Task name (default: traffic_state_pred)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name, e.g., MCSTMamba')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name, e.g., PEMSD8')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the model cache (exp_id directory)')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Specific epoch to load, if not specified will load the best model')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Config file to override settings')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--evaluator_mode', type=str, default='single',
                        help='Evaluator mode (default: single)')
    parser.add_argument('--metrics', type=str, default=None,
                        help='Comma-separated list of metrics to use (default: None, uses config file metrics)')
    
    # MCSTMamba-specific parameters (can be auto-detected from checkpoint)
    parser.add_argument('--d_state', type=int, default=None,
                       help='State dimension for Mamba models')
    parser.add_argument('--d_conv', type=int, default=None,
                       help='Convolution dimension for Mamba models')
    parser.add_argument('--expand', type=int, default=None,
                       help='Expansion factor for Mamba models')
    
    # Evaluation options
    parser.add_argument('--evaluate_channels_separately', action='store_true',
                        help='If set, evaluate each output channel separately')
    
    args = parser.parse_args()
    
    # Extract other args to pass to the config
    other_args = {
        'device': f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu',
        'evaluator_mode': args.evaluator_mode
    }
    
    # Add metrics if specified
    if args.metrics:
        other_args['metrics'] = args.metrics.split(',')
    
    # Add model-specific parameters if specified
    if args.d_state:
        other_args['d_state'] = args.d_state
    if args.d_conv:
        other_args['d_conv'] = args.d_conv
    if args.expand:
        other_args['expand'] = args.expand
    
    # Run evaluation
    evaluate_model(
        task=args.task,
        model_name=args.model,
        dataset_name=args.dataset,
        model_dir=args.model_dir,
        epoch=args.epoch,
        config_file=args.config_file,
        other_args=other_args,
        evaluate_channels_separately=args.evaluate_channels_separately
    ) 