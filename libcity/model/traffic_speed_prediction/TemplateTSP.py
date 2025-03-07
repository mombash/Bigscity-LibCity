from logging import getLogger
import torch
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class TemplateTSP(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        """
        Construct the model
        :param config: Configuration dictionary derived from various configurations
        :param data_feature: Necessary data-related features returned from the Dataset class's `get_data_feature()` interface
        """
        # 1. Initialize the parent class (mandatory)
        super().__init__(config, data_feature)
        # 2. Get the desired information from data_feature; note that different models use different Dataset classes, and the content returned by data_feature varies (mandatory)
        # For example, using TrafficStateGridDataset to demonstrate data extraction, the following data can be extracted; unnecessary ones can be omitted
        # **These parameters cannot be obtained from config**
        self._scaler = self.data_feature.get('scaler')  # For data normalization
        self.adj_mx = self.data_feature.get('adj_mx', 1)  # Adjacency matrix
        self.num_nodes = self.data_feature.get('num_nodes', 1)  # Number of grids
        self.feature_dim = self.data_feature.get('feature_dim', 1)  # Input dimension
        self.output_dim = self.data_feature.get('output_dim', 1)  # Output dimension
        self.len_row = self.data_feature.get('len_row', 1)  # Number of rows in the grid
        self.len_column = self.data_feature.get('len_column', 1)  # Number of columns in the grid
        # 3. Initialize log for necessary outputs (mandatory)
        self._logger = getLogger()
        # 4. Initialize device (mandatory)
        self.device = config.get('device', torch.device('cpu'))
        # 5. Initialize the length of input and output time steps (optional)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        # 6. Extract other parameters used from config, mainly for constructing model structure parameters (mandatory)
        # These parameters related to model structure should be placed in libcity/config/model/model_name.json (mandatory)
        # For example: self.blocks = config['blocks']
        # ...
        # 7. Construct the hierarchical structure of the deep model (mandatory)
        # For example: Using a simple RNN: self.rnn = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, batch):
        """
        Call the model to compute the output corresponding to this batch input, an interface that nn.Module must implement
        :param batch: Input data, a dictionary-like object, can be accessed like a dictionary
        :return:
        """
        # 1. Extract data; assume there are 4 types of data in the dictionary: X, y, X_ext, y_ext
        # Generally, only input data is needed, e.g., X, X_ext, because this function is used to compute outputs
        # The feature dimension of the model input data should equal self.feature_dim
        # For example: y = batch['y'] / X_ext = batch['X_ext'] / y_ext = batch['y_ext']]
        # 2. Calculate the model's output based on the input data
        # The feature dimension of the model's output should equal self.output_dim
        # The other dimensions of the model's output should be consistent with batch['y'], only the feature dimension may differ (because batch['y'] may contain some external features)
        # If the model performs single-step prediction, and batch['y'] contains multi-step data, the time dimensions may also differ
        # For example: outputs = self.model(x)
        # 3. Return the output result
        # For example: return outputs

    def calculate_loss(self, batch):
        """
        Input a batch of data and return the loss for this batch during the training process, which requires defining a loss function.
        :param batch: Input data, a dictionary-like object, can be accessed like a dictionary
        :return: training loss (tensor)
        """
        # 1. Extract the ground truth
        y_true = batch['y']
        # 2. Extract the predicted values
        y_predicted = self.predict(batch)
        # 3. Use self._scaler to reverse normalize the ground truth and predicted values that have been normalized (mandatory)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # 4. Call the loss function to calculate the error between the ground truth and predicted values
        # Common loss functions are defined in libcity/model/loss.py
        # If the model source code uses one of the losses, it can be called directly; for example, using MSE:
        res = loss.masked_mse_torch(y_predicted, y_true)
        # If the loss function used in the model source code is not in loss.py, it needs to be implemented by yourself
        # ... (custom loss function)
        # 5. Return the loss result
        return res

    def predict(self, batch):
        """
        Input a batch of data and return the corresponding predicted values, generally should be the result of **multi-step prediction**
        Generally, the above-defined forward() method will be called
        :param batch: Input data, a dictionary-like object, can be accessed like a dictionary
        :return: predict result of this batch (tensor)
        """
        # If the result of self.forward() meets the requirements, it can be returned directly
        # If it does not meet the requirements, for example, if self.forward() performs single-time-step prediction, but the model training uses multi-step data for each batch,
        # then refer to the predict() function in libcity/model/traffic_speed_prediction/STGCN.py for multi-step prediction
        # The principle of multi-step prediction is: first perform one-step prediction, use the result of one-step prediction for two-step prediction, **instead of using the true value of one-step prediction for two-step prediction!**
        # For example, if the result of self.forward() meets the requirements:
        return self.forward(batch)
