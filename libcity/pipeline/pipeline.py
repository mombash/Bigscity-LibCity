import os
# from ray import tune # Commented out to avoid ray dependency
# from ray.tune.suggest.hyperopt import HyperOptSearch # Commented out to avoid ray dependency
# from ray.tune.suggest.bayesopt import BayesOptSearch # Commented out to avoid ray dependency
# from ray.tune.suggest.basic_variant import BasicVariantGenerator # Commented out to avoid ray dependency
# from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, MedianStoppingRule # Commented out to avoid ray dependency
# from ray.tune.suggest import ConcurrencyLimiter # Commented out to avoid ray dependency
import json
import torch
import random
import shutil
import glob
import re
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed
import time


def extract_loss_data(log_content):
    """
    Extract epoch, training loss, and validation loss data from log content.
    
    Args:
        log_content (str): Content of the log file
        
    Returns:
        tuple: (epochs, train_losses, val_losses, learning_rates) lists
    """
    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # Regular expression to match the epoch summary lines
    # Example: "Epoch [0/5] train_loss: 87.4590, val_loss: 75.8063, lr: 0.001000, 85.43s"
    pattern = r'Epoch \[(\d+)/\d+\] train_loss: ([\d\.]+), val_loss: ([\d\.]+), lr: ([\d\.]+)'
    
    for line in log_content.split('\n'):
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


def generate_loss_curve(log_content, output_dir, exp_id, model_name, dataset_name):
    """
    Generate a visualization of training and validation loss curves.
    
    Args:
        log_content (str): Content of the log file
        output_dir (str): Directory to save the visualization
        exp_id (str): Experiment ID for the plot title
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
    
    Returns:
        list: Paths to the generated visualization files
    """
    epochs, train_losses, val_losses, learning_rates = extract_loss_data(log_content)
    
    if not epochs:
        return []
    
    output_files = []
    
    # Create the main figure for loss curves
    plt.figure(figsize=(12, 6))
    
    # Plot both training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-^', label='Validation Loss')
    
    # Set labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} on {dataset_name}\nTraining and Validation Loss')
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
    output_files.append(output_file)
    
    # Create a second figure for log-scale view if needed (useful for exponential decay)
    if min(train_losses) > 0 and min(val_losses) > 0:  # Ensure positive values for log scale
        plt.figure(figsize=(10, 6))
        plt.semilogy(epochs, train_losses, 'b-o', label='Training Loss')
        plt.semilogy(epochs, val_losses, 'r-^', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title(f'{model_name} on {dataset_name}\nLoss Curves (Log Scale)')
        plt.grid(True, alpha=0.3, which='both')
        plt.legend()
        
        # Mark minimum validation loss point
        if val_losses:
            plt.axvline(x=min_val_epoch, color='g', linestyle='--', alpha=0.5)
            plt.plot(min_val_epoch, min_val_loss, 'go', markersize=10)
        
        # Save the log-scale figure
        log_output_file = os.path.join(output_dir, 'loss_curve_log_scale.png')
        plt.savefig(log_output_file, dpi=150, bbox_inches='tight')
        plt.close()
        output_files.append(log_output_file)
    
    return output_files


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Create a descriptive experiment ID with model name, dataset, and timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_id = f"{model_name}_{dataset_name}_{timestamp}"
        config['exp_id'] = exp_id
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)
    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # 加载执行器
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)
    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    # 训练
    if train or not os.path.exists(model_cache_file):
        executor.train(train_data, valid_data)
        if saved_model:
            executor.save_model(model_cache_file)
    else:
        executor.load_model(model_cache_file)
    # 评估，评估结果将会放在 cache/evaluate_cache 下
    executor.evaluate(test_data)
    
    # Copy log file to the cache directory for easier reference
    # Find the log file corresponding to this experiment
    log_dir = './libcity/log'
    cache_dir = './libcity/cache'
    # Ensure cache logs directory exists
    logs_cache_dir = os.path.join(cache_dir, exp_id, 'logs')
    os.makedirs(logs_cache_dir, exist_ok=True)
    
    # Get log files for this experiment
    log_files = glob.glob(f"{log_dir}/{exp_id}*.log")
    for log_file in log_files:
        # Copy log file to cache directory
        basename = os.path.basename(log_file)
        target_file = os.path.join(logs_cache_dir, basename)
        shutil.copy2(log_file, target_file)
        logger.info(f"Copied log file to cache: {target_file}")
    
    # Create a more user-friendly summary file with key metrics
    try:
        if log_files:
            with open(log_files[0], 'r') as f:
                log_content = f.read()
                
            # Extract key metrics
            summary_file = os.path.join(logs_cache_dir, 'training_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(f"Training Summary for Experiment: {exp_id}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write("-" * 50 + "\n\n")
                
                # Extract and write training epochs info
                training_lines = [line for line in log_content.split('\n') if 'train_loss' in line and 'val_loss' in line]
                if training_lines:
                    f.write("TRAINING PROGRESS:\n")
                    for line in training_lines:
                        f.write(line + "\n")
                    f.write("\n")
                
                # Extract and write evaluation metrics
                eval_lines = [line for line in log_content.split('\n') if 'Evaluate inference' in line]
                if eval_lines:
                    f.write("EVALUATION RESULTS:\n")
                    for line in eval_lines:
                        f.write(line + "\n")
                    
                    metrics_lines = [line for line in log_content.split('\n') if any(metric in line for metric in ['MAE', 'MAPE', 'MSE', 'RMSE'])]
                    for line in metrics_lines[-10:]:  # Last 10 metrics lines
                        f.write(line + "\n")
            
            logger.info(f"Created training summary: {summary_file}")
            
            # Generate and save loss curve visualizations
            try:
                viz_files = generate_loss_curve(log_content, logs_cache_dir, exp_id, model_name, dataset_name)
                for viz_file in viz_files:
                    logger.info(f"Generated visualization: {viz_file}")
            except Exception as e:
                logger.warning(f"Error generating loss curve visualizations: {str(e)}")
                
    except Exception as e:
        logger.warning(f"Error creating training summary: {str(e)}")


def parse_search_space(space_file):
    search_space = {}
    if os.path.exists('./{}.json'.format(space_file)):
        with open('./{}.json'.format(space_file), 'r') as f:
            paras_dict = json.load(f)
            for name in paras_dict:
                paras_type = paras_dict[name]['type']
                if paras_type == 'uniform':
                    # name type low up
                    try:
                        search_space[name] = tune.uniform(paras_dict[name]['lower'], paras_dict[name]['upper'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing uniform type.')
                elif paras_type == 'randn':
                    # name type mean sd
                    try:
                        search_space[name] = tune.randn(paras_dict[name]['mean'], paras_dict[name]['sd'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing randn type.')
                elif paras_type == 'randint':
                    # name type lower upper
                    try:
                        if 'lower' not in paras_dict[name]:
                            search_space[name] = tune.randint(paras_dict[name]['upper'])
                        else:
                            search_space[name] = tune.randint(paras_dict[name]['lower'], paras_dict[name]['upper'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing randint type.')
                elif paras_type == 'choice':
                    # name type list
                    try:
                        search_space[name] = tune.choice(paras_dict[name]['list'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing choice type.')
                elif paras_type == 'grid_search':
                    # name type list
                    try:
                        search_space[name] = tune.grid_search(paras_dict[name]['list'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing grid_search type.')
                else:
                    raise TypeError('The space file does not meet the format requirements,\
                            when parsing an undefined type.')
    else:
        raise FileNotFoundError('The space file {}.json is not found. Please ensure \
            the config file is in the root dir and is a txt.'.format(space_file))
    return search_space


def hyper_parameter(task=None, model_name=None, dataset_name=None, config_file=None, space_file=None,
                    scheduler=None, search_alg=None, other_args=None, num_samples=5, max_concurrent=1,
                    cpu_per_trial=1, gpu_per_trial=1):
    """ Use Ray tune to hyper parameter tune

    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        space_file(str): the file which specifies the parameter search space
        scheduler(str): the trial sheduler which will be used in ray.tune.run
        search_alg(str): the search algorithm
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    experiment_config = ConfigParser(task, model_name, dataset_name, config_file=config_file,
                                     other_args=other_args)
    # exp_id
    exp_id = experiment_config.get('exp_id', None)
    if exp_id is None:
        # Create a descriptive experiment ID with model name, dataset, and timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_id = f"hyper_{model_name}_{dataset_name}_{timestamp}"
        experiment_config['exp_id'] = exp_id
    # logger
    logger = get_logger(experiment_config)
    logger.info('Begin ray-tune, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(experiment_config.config)
    # check space_file
    if space_file is None:
        logger.error('the space_file should not be None when hyperparameter tune.')
        exit(0)
    # seed
    seed = experiment_config.get('seed', 0)
    set_random_seed(seed)
    # parse space_file
    search_sapce = parse_search_space(space_file)
    # load dataset
    dataset = get_dataset(experiment_config)
    # get train valid test data
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    def train(config, checkpoint_dir=None, experiment_config=None,
              train_data=None, valid_data=None, data_feature=None):
        """trainable function which meets ray tune API

        Args:
            config (dict): A dict of hyperparameter.
        """
        # modify experiment_config
        for key in config:
            if key in experiment_config:
                experiment_config[key] = config[key]
        experiment_config['hyper_tune'] = True
        logger = get_logger(experiment_config)
        # exp_id
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        task = experiment_config.get('task', 'unknown')
        model_name = experiment_config.get('model', 'unknown')
        dataset_name = experiment_config.get('dataset', 'unknown')
        exp_id = f"hyper_trial_{model_name}_{dataset_name}_{timestamp}"
        experiment_config['exp_id'] = exp_id
        logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                    format(str(task), str(model_name), str(dataset_name), str(exp_id)))
        logger.info('running parameters: ' + str(config))
        # load model
        model = get_model(experiment_config, data_feature)
        # load executor
        executor = get_executor(experiment_config, model, data_feature)
        # checkpoint by ray tune
        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, 'checkpoint')
            executor.load_model(checkpoint)
        # train
        executor.train(train_data, valid_data)

    # init search algorithm and scheduler
    if search_alg == 'BasicSearch':
        algorithm = BasicVariantGenerator()
    elif search_alg == 'BayesOptSearch':
        algorithm = BayesOptSearch(metric='loss', mode='min')
        # add concurrency limit
        algorithm = ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
    elif search_alg == 'HyperOpt':
        algorithm = HyperOptSearch(metric='loss', mode='min')
        # add concurrency limit
        algorithm = ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
    else:
        raise ValueError('the search_alg is illegal.')
    if scheduler == 'FIFO':
        tune_scheduler = FIFOScheduler()
    elif scheduler == 'ASHA':
        tune_scheduler = ASHAScheduler()
    elif scheduler == 'MedianStoppingRule':
        tune_scheduler = MedianStoppingRule()
    else:
        raise ValueError('the scheduler is illegal')
    # ray tune run
    ensure_dir('./libcity/cache/hyper_tune')
    result = tune.run(tune.with_parameters(train, experiment_config=experiment_config, train_data=train_data,
                      valid_data=valid_data, data_feature=data_feature),
                      resources_per_trial={'cpu': cpu_per_trial, 'gpu': gpu_per_trial}, config=search_sapce,
                      metric='loss', mode='min', scheduler=tune_scheduler, search_alg=algorithm,
                      local_dir='./libcity/cache/hyper_tune', num_samples=num_samples)
    best_trial = result.get_best_trial("loss", "min", "last")
    logger.info("Best trial config: {}".format(best_trial.config))
    logger.info("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    # save best
    best_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    model_state, optimizer_state = torch.load(best_path)
    model_cache_file = './libcity/cache/{}/model_cache/{}_{}.m'.format(
        exp_id, model_name, dataset_name)
    ensure_dir('./libcity/cache/{}/model_cache'.format(exp_id))
    torch.save((model_state, optimizer_state), model_cache_file)


def objective_function(task=None, model_name=None, dataset_name=None, config_file=None,
                       saved_model=True, train=True, other_args=None, hyper_config_dict=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
        hyper_config_dict(dict): the dict of hyperparameter combinatio
    """
    if other_args is None:
        other_args = {}
    if hyper_config_dict is not None:
        other_args.update(hyper_config_dict)
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)
    # Make a new experiment ID with descriptive name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_id = f"obj_fn_{model_name}_{dataset_name}_{timestamp}"
    config['exp_id'] = exp_id
    # logger
    logger = get_logger(config)
    logger.info('Begin pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    dataset = get_dataset(config)
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()

    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)
    best_valid_score = executor.train(train_data, valid_data)
    test_result = executor.evaluate(test_data)

    return {
        'best_valid_score': best_valid_score,
        'test_result': test_result
    }
