"""
This file contains the class which constructs the TensorFlow graph of ForecastNet and provides a function for
forecasting.

Paper:
"ForecastNet: A Time-Variant Deep Feed-Forward Neural Network Architecture for Multi-Step-Ahead Time-Series Forecasting"
by Joel Janek Dabrowski, YiFan Zhang, and Ashfaqur Rahman
Link to the paper: https://arxiv.org/abs/2002.04155
"""


import numpy as np
import torch
from torch import nn

from .dense_forecastnet import ForecastNetDenseModel2
from .data_helpers import format_input, format_shape


class ForecastNet:
    """
    Class for ForecastNet.
    """

    def __init__(self, in_seq_length, out_seq_length, input_dim, 
                 n_layers, hidden_dim, output_dim, #model_type='dense2', 
                 batch_size=1, max_epochs=100, learning_rate=0.0001, 
                 save_file='./forecastnet.pt'):
        """
        Constructor
        :param in_seq_length: Sequence length of the inputs.
        :param out_seq_length: Sequence length of the outputs.
        :param input_dim: Dimension of the inputs
        :param hidden_dim: Dimension of the hidden units
        :param output_dim: Dimension of the outputs
        :param model_type: Use 'dense' for a two layered densely connected hidden cell and Mixture Density network outputs.
                           Use 'conv' for the convolutional hidden cell and Mixture Density network outputs.
                           Use 'dense2' for a two layered densely connected hidden cell and linear outputs.
                           Use 'conv2' for the convolutional hidden cell and linear outputs.
        :param batch_size: Batch size to use during training. Default: 1
        :param max_epochs: Number of epochs to train over: Default: 100
        :param learning_rate: Learning rate for the Adam algorithm. Default: 0.0001
        :param save_file: Path and filename to save the model to. Default: './forecastnet.pt'
        """
        # Number of sequence steps == number of RNN cells: 28 pixels in the column
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # self.model_type = model_type
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.save_file = save_file

        # Use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create the ForecastNet model
        # if model_type == 'dense':
        #     self.model = ForecastNetDenseModel(self.input_dim, self.hidden_dim, self.output_dim, self.in_seq_length, self.out_seq_length, self.device)
        # elif model_type == 'conv':
        #     self.model = ForecastNetConvModel(self.input_dim, self.hidden_dim, self.output_dim, self.in_seq_length, self.out_seq_length, self.device)
        # elif model_type == 'dense2':
        #     self.model = ForecastNetDenseModel2(self.input_dim, self.hidden_dim, self.output_dim, self.in_seq_length, self.out_seq_length, self.device)
        # elif model_type == 'conv2':
        #     self.model = ForecastNetConvModel2(self.input_dim, self.hidden_dim, self.output_dim, self.in_seq_length, self.out_seq_length, self.device)
        self.model = ForecastNetDenseModel2(
            self.input_dim, self.n_layers, self.hidden_dim, self.output_dim, 
            self.in_seq_length, self.out_seq_length, self.device, 
            )

        # Use multiple GPUS
        if torch.cuda.device_count() > 1:
            print('Using %d GPUs'%(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)

        # Send model to the selected device
        self.model.to(self.device)

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # print('Trainable variables = ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        # print('')

    def forecast(self, test_x, load_checkpoint=False):
        """
        Perform a forecast given an input test dataset.
        :param test_x: Input test data in the form [in_seq_length, batch_size, input_dim]
        :return: y_hat: The sampled forecast as a numpy array in the form [out_seq_length, batch_size, output_dim]
        :return: mu: The mean forecast as a numpy array in the form [out_seq_length, batch_size, output_dim]
                     (Only returned if the model is 'dense' or 'conv')
        :return: sigma: The standard deviation forecast as a numpy array in the form [out_seq_length, batch_size, output_dim]
                        (Only returned if the model is 'dense' or 'conv')
        """
        # n_tests = test_x.shape[1]

        self.model.eval()

        if load_checkpoint:
            # Load model parameters
            checkpoint = torch.load(self.save_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        with torch.no_grad():

            if type(test_x) is np.ndarray:
                test_x = torch.from_numpy(test_x).type(torch.FloatTensor)

            # # Format the inputs
            # test_x = format_input(test_x)
            # # # Dummy output
            # # empty_y = torch.empty((self.out_seq_length, n_tests, self.output_dim))

            # test_x = test_x.to(self.device)
            # # empty_y = empty_y.to(self.device)

            # Compute the forecast
            # if self.model_type == 'dense' or self.model_type == 'conv':
            #     y_hat, mu, sigma = self.model(test_x, empty_y, is_training=False)
            # elif self.model_type == 'dense2' or self.model_type == 'conv2':
            #     y_hat = self.model(test_x, empty_y, is_training=False)
            y_hat = self.model(test_x, is_training=False)

        # if self.model_type == 'dense' or self.model_type == 'conv':
        #     return y_hat.cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy()
        # elif self.model_type == 'dense2' or self.model_type == 'conv2':
        #     return y_hat.cpu().numpy()
        return y_hat.cpu().numpy()
        
    def forecast_2d(self, test_x, load_checkpoint=True):
        """
        Perform a forecast given an input test dataset and output in 2d.
        :param test_x: Input test data in the form [batch_size, in_seq_length*input_dim]
        """
        # 2d array to 3d array
        test_x = test_x.reshape((test_x.shape[0], self.in_seq_length, self.input_dim))
        test_x = test_x.transpose((1, 0, 2))

        # Return 2d arrays
        # if self.model_type == 'dense' or self.model_type == 'conv':
        #     y_hat, mu, sigma = self.forecast(test_x)
        #     return format_shape(y_hat), format_shape(mu), format_shape(sigma)
        # elif self.model_type == 'dense2' or self.model_type == 'conv2':
        #     y_hat = self.forecast(test_x)
        #     return format_shape(y_hat)
        y_hat = self.forecast(test_x, load_checkpoint)
        return format_shape(y_hat)
        
