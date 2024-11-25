

import torch
import torch.nn as nn
from torch.nn import RNN
# https://colah.github.io/posts/2015-08-Understanding-GRUs/
# https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
# https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# https://www.kaggle.com/code/columbine/seq2seq-pytorch


# self=GRUDenseModel()
# self=forecast_model.model
class RNNDenseModel(nn.Module):
    """
    Class for the densely connected hidden cells version of the model
    """
    def __init__(
            self, input_dim, n_layers, hidden_dim, output_dim, dropout, bidirectional, 
            in_seq_length, out_seq_length, device, 
            ):
        """
        Constructor
        :param input_dim: Dimension of the inputs
        :param hidden_dim: Number of hidden units
        :param output_dim: Dimension of the outputs
        :param in_seq_length: Length of the input sequence
        :param out_seq_length: Length of the output sequence
        """
        super(RNNDenseModel, self).__init__()

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.device = device

        # Initialise layers
        self.model = RNN( # from nn.RNN
            input_size=input_dim, 
            num_layers=n_layers, 
            hidden_size=hidden_dim, 
            batch_first=False, 
            dropout=dropout, 
            bidirectional=bidirectional, 
        )

        self.output_layer = nn.Linear(in_seq_length*hidden_dim, output_dim)
    def forward(self, input, target=None, is_training=False):
        """
        Forward propagation of the dense GRU model
        :param input: Input data in the form [in_seq_length, batch_size, input_dim]
        :param target: Target data in the form [out_seq_length, batch_size, output_dim]
        :param is_training: If true, use target data for training, else use the previous output.
        :return: outputs: Forecast outputs in the form [decoder_seq_length, batch_size, input_dim]
        """
        # me
        self.model.flatten_parameters()

        # Initialise outputs
        outputs = torch.zeros((self.out_seq_length, input.shape[1], self.output_dim)).to(self.device)
        
        for i in range(self.out_seq_length):
            if i==0:
                # First input
                output, hidden_state = self.model(input)
            else:
                # Calculate the output
                output, hidden_state = self.model(input, hidden_state)

            outputs[i,:,:] = self.output_layer(output.permute(1, 0, 2).reshape(input.shape[1], -1))

        return outputs
