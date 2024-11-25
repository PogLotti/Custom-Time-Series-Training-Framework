

import numpy as np
import torch


def format_shape(arr):
    """
    :param X: array with shape [in_seq_length, n_batches, input_dim]
    :return: array with shape [n_batches, in_seq_length * input_dim]
    """
    return arr.transpose((1, 0, 2)).reshape((arr.shape[1], -1))


def format_input(input):
    """
    Format the input array by combining the time and input dimension of the input.
    That is: reshape from [in_seq_length, n_batches, input_dim] to [n_batches, in_seq_length * input_dim]
    :param input: input tensor with shape [in_seq_length, n_batches, input_dim]
    :return: input tensor reshaped to [n_batches, in_seq_length * input_dim]
    """
    if type(input) is np.ndarray:
        input = torch.from_numpy(input).type(torch.FloatTensor)

    in_seq_length, batch_size, input_dim = input.shape
    input_reshaped = input.permute(1, 0, 2)
    input_reshaped = torch.reshape(input_reshaped, (batch_size, -1))
    return input_reshaped


