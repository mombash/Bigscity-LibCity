#!/usr/bin/env python
import os
import sys
import glob
import shutil
import re
import matplotlib.pyplot as plt
import numpy as np

def copy_logs_to_cache(exp_id=None):
    """
    Copies log files for a specific experiment ID to the corresponding cache directory.
    
    Args:
        exp_id (str): Optional experiment ID filter. If None, will copy all logs to corresponding cache dirs.
    """
    # Find log files
    log_dir = './libcity/log'
    cache_dir = './libcity/cache'
    
    # Make sure directories exist
    if not os.path.exists(log_dir):
        print(f"Error: Log directory {log_dir} does not exist.")
        return
    
    if not os.path.exists(cache_dir):
        print(f"Error: Cache directory {cache_dir} does not exist.")
        return
    
    # Get log files that match the experiment ID
    if exp_id:
        log_files = glob.glob(f"{log_dir}/{exp_id}*log")
    else:
        log_files = glob.glob(f"{log_dir}/*.log")
    
    if not log_files:
        print(f"No log files found for {'experiment ' + exp_id if exp_id else 'any experiment'}.")
        return
    
    # Process each log file
    for log_file in log_files:
        # Extract experiment ID from the log file name
        basename = os.path.basename(log_file)
        # The exp_id is the part before the first hyphen
        file_exp_id = basename.split('-')[0]
        
        # Find the corresponding cache directory
        cache_dirs = glob.glob(f"{cache_dir}/{file_exp_id}")
        
        if not cache_dirs:
            print(f"No cache directory found for experiment ID {file_exp_id}")
            continue
        
        # The cache directory should be a direct match to the experiment ID
        exp_cache_dir = cache_dirs[0]
        
        # Create logs directory in the cache if it doesn't exist
        logs_cache_dir = os.path.join(exp_cache_dir, 'logs')
        os.makedirs(logs_cache_dir, exist_ok=True)
        
        # Copy the log file to the cache directory
        target_file = os.path.join(logs_cache_dir, basename)
        shutil.copy2(log_file, target_file)
        print(f"Copied {log_file} to {target_file}")
        
        # Generate and save loss curve visualization
        try:
            generate_loss_curve(log_file, logs_cache_dir, file_exp_id)
        except Exception as e:
            print(f"Error generating loss curve: {str(e)}")

def extract_loss_data(log_file):
    """
    Extract epoch, training loss, and validation loss data from a log file.
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        tuple: (epochs, train_losses, val_losses) lists
    """
    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # Regular expression to match the epoch summary lines
    # Example: "Epoch [0/5] train_loss: 87.4590, val_loss: 75.8063, lr: 0.001000, 85.43s"
    pattern = r'Epoch \[(\d+)/\d+\] train_loss: ([\d\.]+), val_loss: ([\d\.]+), lr: ([\d\.]+)'
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                lr = float(match.group(4))
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                learning_rates.append(lr)
    
    return epochs, train_losses, val_losses, learning_rates

def generate_loss_curve(log_file, output_dir, exp_id):
    """
    Generate a visualization of training and validation loss curves.
    
    Args:
        log_file (str): Path to the log file
        output_dir (str): Directory to save the visualization
        exp_id (str): Experiment ID for the plot title
    """
    epochs, train_losses, val_losses, learning_rates = extract_loss_data(log_file)
    
    if not epochs:
        print(f"No training data found in log file: {log_file}")
        return
    
    # Create the main figure for loss curves
    plt.figure(figsize=(12, 6))
    
    # Plot both training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-^', label='Validation Loss')
    
    # Set labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss\n{exp_id}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Find minimum validation loss point
    if val_losses:
        min_val_idx = val_losses.index(min(val_losses))
        min_val_epoch = epochs[min_val_idx]
        min_val_loss = val_losses[min_val_idx]
        
        # Mark the best model point
        plt.axvline(x=min_val_epoch, color='g', linestyle='--', alpha=0.5)
        plt.plot(min_val_epoch, min_val_loss, 'go', markersize=10)
        plt.annotate(f'Best: {min_val_loss:.4f}', 
                     xy=(min_val_epoch, min_val_loss),
                     xytext=(min_val_epoch + 0.2, min_val_loss * 1.1),
                     arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.7))
    
    # Plot learning rate in a separate subplot
    if learning_rates:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, learning_rates, 'g-o', label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Adjust layout and save the figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated loss curve visualization: {output_file}")
    
    # Create a second figure for log-scale view if needed (useful for exponential decay)
    if min(train_losses) > 0 and min(val_losses) > 0:  # Ensure positive values for log scale
        plt.figure(figsize=(10, 6))
        plt.semilogy(epochs, train_losses, 'b-o', label='Training Loss')
        plt.semilogy(epochs, val_losses, 'r-^', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title(f'Training and Validation Loss (Log Scale)\n{exp_id}')
        plt.grid(True, alpha=0.3, which='both')
        plt.legend()
        
        # Save the log-scale figure
        log_output_file = os.path.join(output_dir, 'loss_curve_log_scale.png')
        plt.savefig(log_output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Generated log-scale loss curve: {log_output_file}")

if __name__ == "__main__":
    # Check if an experiment ID was provided
    exp_id = sys.argv[1] if len(sys.argv) > 1 else None
    copy_logs_to_cache(exp_id)
    print("Done!") 