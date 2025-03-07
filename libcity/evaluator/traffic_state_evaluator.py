import os
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from libcity.utils import ensure_dir
from libcity.model import loss
from logging import getLogger
from libcity.evaluator.abstract_evaluator import AbstractEvaluator
import torch


class TrafficStateEvaluator(AbstractEvaluator):

    def __init__(self, config):
        self.metrics = config.get('metrics', ['MAE'])  # 评估指标, 是一个 list
        self.allowed_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'masked_MAE', 'masked_MSE', 'masked_RMSE', 'masked_MAPE',
                                'R2', 'EVAR']
        self.save_modes = config.get('save_mode', ['csv', 'json'])
        if not isinstance(self.save_modes, list):
            self.save_modes = [self.save_modes]
        self.mode = config.get('evaluator_mode', 'single')  # or average
        self.mask_val = config.get('mask_val', None)
        self.config = config
        self.len_timeslots = 0
        self.result = {}  # 每一种指标的结果
        self.intermediate_result = {}  # 每一种指标每一个batch的结果
        self.visualization_path = config.get('visualization_path', './visualization')
        ensure_dir(self.visualization_path)
        self._check_config()
        self._logger = getLogger()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in TrafficStateEvaluator'.format(str(metric)))

    def visualize_prediction(self, batch):
        """
        Visualize prediction vs ground truth for a random sample from the batch
        
        Args:
            batch(dict): Input data containing y_true and y_pred
        """
        y_true = batch['y_true']  # tensor
        y_pred = batch['y_pred']  # tensor
        
        # Randomly select one sample from the batch
        batch_size = y_true.shape[0]
        random_idx = torch.randint(0, batch_size, (1,)).item()
        sample_true = y_true[random_idx].cpu().numpy()
        sample_pred = y_pred[random_idx].cpu().numpy()
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        time_steps = range(sample_true.shape[0])
        
        # Plot for each feature dimension
        for dim in range(sample_true.shape[-1]):
            plt.subplot(sample_true.shape[-1], 1, dim + 1)
            plt.plot(time_steps, sample_true[:, 0, dim], 'b-', label='Ground Truth')
            plt.plot(time_steps, sample_pred[:, 0, dim], 'r--', label='Prediction')
            plt.legend()
            plt.title(f'Feature Dimension {dim}')
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(self.visualization_path, f'prediction_viz_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()
        
        self._logger.info(f'Visualization saved to {save_path}')

    def collect(self, batch):
        """
        收集一 batch 的评估输入，随机选择一个样本进行评估

        Args:
            batch(dict): 输入数据，字典类型，包含两个Key:(y_true, y_pred):
                batch['y_true']: (num_samples/batch_size, timeslots, ..., feature_dim)
                batch['y_pred']: (num_samples/batch_size, timeslots, ..., feature_dim)
        """
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        y_true = batch['y_true']  # tensor
        y_pred = batch['y_pred']  # tensor
        if y_true.shape != y_pred.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")
        
        # # Randomly select one sample from the batch
        # batch_size = y_true.shape[0]
        # random_idx = torch.randint(0, batch_size, (1,)).item()
        # y_true = y_true[random_idx:random_idx+1]  # Keep dimension with size 1
        # y_pred = y_pred[random_idx:random_idx+1]
        
        self.len_timeslots = y_true.shape[1]
        for i in range(1, self.len_timeslots + 1):
            for metric in self.metrics:
                if metric + '@' + str(i) not in self.intermediate_result:
                    self.intermediate_result[metric + '@' + str(i)] = []
        if self.mode.lower() == 'average':  # 前i个时间步的平均loss
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    if metric == 'masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, :i], y_true[:, :i], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, :i], y_true[:, :i], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, :i], y_true[:, :i], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, :i], y_true[:, :i], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'R2':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.r2_score_torch(y_pred[:, :i], y_true[:, :i]).item())
                    elif metric == 'EVAR':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.explained_variance_score_torch(y_pred[:, :i], y_true[:, :i]).item())
        elif self.mode.lower() == 'single':  # 第i个时间步的loss
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    if metric == 'masked_MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'masked_MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1], 0,
                                                  mask_val=self.mask_val).item())
                    elif metric == 'MAE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'MSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'RMSE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'MAPE':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            loss.masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                    elif metric == 'R2':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            # loss.r2_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                            loss.r2_score_torch(y_pred[:, i - 1], y_true[:, i - 1]))
                    elif metric == 'EVAR':
                        self.intermediate_result[metric + '@' + str(i)].append(
                            # loss.explained_variance_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
                            loss.explained_variance_score_torch(y_pred[:, i - 1], y_true[:, i - 1]))
                        
        else:
            raise ValueError('Error parameter evaluator_mode={}, please set `single` or `average`.'.format(self.mode))

        self.visualize_prediction(batch)

    def evaluate(self):
        """
        返回之前收集到的所有 batch 的评估结果
        """
        for i in range(1, self.len_timeslots + 1):
            for metric in self.metrics:
                self.result[metric + '@' + str(i)] = sum(self.intermediate_result[metric + '@' + str(i)]) / \
                                                     len(self.intermediate_result[metric + '@' + str(i)])
        return self.result

    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
        """
        self._logger.info('Note that you select the {} mode to evaluate!'.format(self.mode))
        self.evaluate()
        ensure_dir(save_path)
        if filename is None:  # 使用时间戳
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model'] + '_' + self.config['dataset']

        if 'json' in self.save_modes:
            self._logger.info('Evaluate result is ' + json.dumps(self.result))
            with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') as f:
                json.dump(self.result, f)
            self._logger.info('Evaluate result is saved at ' +
                              os.path.join(save_path, '{}.json'.format(filename)))

        if 'csv' in self.save_modes:
            dataframe = {}
            for metric in self.metrics:
                dataframe[metric] = []
            for i in range(1, self.len_timeslots + 1):
                for metric in self.metrics:
                    dataframe[metric].append(self.result[metric + '@' + str(i)])
            
            # Convert to DataFrame and save
            df = pd.DataFrame(dataframe)
            csv_path = os.path.join(save_path, '{}.csv'.format(filename))
            df.to_csv(csv_path, index=False)
            self._logger.info('Evaluate result is saved at ' + csv_path)
        return dataframe

    def clear(self):
        """
        清除之前收集到的 batch 的评估信息，适用于每次评估开始时进行一次清空，排除之前的评估输入的影响。
        """
        self.result = {}
        self.intermediate_result = {}
