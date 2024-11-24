import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from copy import copy
from scipy.fft import fft, fftfreq
from sklearn import cluster
import matplotlib.pyplot as plt

def save(var, dir, verbose=False):
    with open(dir, "wb") as f:  # Python 3: open(..., "wb")
        pickle.dump(var, f)
    
    if verbose:
        print(f"\nSaved to {dir}")

def load(dir):
    with open(dir, "rb") as f:  # Python 3: open(..., "rb")
        var = pickle.load(f)
    
    return var

def split_data(n_samples, valid_ratio=0.1, test_ratio=0.1):
    """
    Split the data samples into training set, validation set and test set
    :param n_samples: total number of data samples
    :param valid_ratio: Ratio of validation set
    :param test_ratio: Ratio of test set
    :return: Index list of training set, validation set and test set
    """
    idx = np.arange(n_samples, dtype=int)
    valid_idx = np.linspace(idx[0], idx[-1], int(valid_ratio*n_samples), 
                            endpoint=True, dtype=int).tolist()
    valid_idx.sort()
    test_idx = idx[-int(test_ratio*n_samples):].tolist()
    test_idx.sort()
    train_idx = list(set(idx)-set(valid_idx)-set(test_idx))
    train_idx.sort()

    return train_idx, valid_idx, test_idx

class Preprocessor_x():
    def fit(self, dataset, axis=0):

        self.axis = axis
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(dataset)

    def transform(self, dataset):

        dataset = self.scaler.transform(dataset)
        return dataset
        
    def fit_transform(self, dataset, axis=0):

        self.fit(dataset, axis=axis)
        dataset = self.transform(dataset)
        return dataset

    def inverse_transform(self, dataset):

        dataset = self.scaler.inverse_transform(dataset)
        return dataset

class Preprocessor_y():
    def fit(self, dataset, axis=0):

        self.axis = axis
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(dataset)

    def transform(self, dataset):

        dataset = self.scaler.transform(dataset)
        return dataset

    def fit_transform(self, dataset, axis=0):

        self.fit(dataset, axis=axis)
        dataset = self.transform(dataset)
        return dataset

    def inverse_transform(self, dataset):

        dataset = self.scaler.inverse_transform(dataset)
        return dataset
    
class DataHolder:
    def __init__(self, config):
        self.config = config

    def transform(self, x, y, train_idx, axis=0):
        preprocessor_x = Preprocessor_x() # create object
        preprocessor_x.fit(x[train_idx, :], axis) 
        x = preprocessor_x.transform(x)

        preprocessor_y = Preprocessor_y()
        preprocessor_y.fit(y[train_idx, :], axis)
        y = preprocessor_y.transform(y)

        return x, preprocessor_x, y, preprocessor_y
        
    def inverse_transform(self, arr, item: str, idx_dict: dict(np.array([]))): 
        arr = copy(np.array(arr))
        for file, idx in idx_dict.items():
            if len(idx)==0:
                continue
            preprocessor = self.data[file][item]
            arr[idx, :] = preprocessor.inverse_transform(arr[idx, :])
        return arr    

    def add(self):

        self.data = dict()
        self.files = [f.name for f in self.config.data_path.glob(f"{self.config.dataset_prefix}*")]

        x_features = self.config.features  
        y_features = self.config.version

        print(f"Loading dataset: {self.files}")
        for file in self.files:
            file_path = Path().joinpath(self.config.data_path, file)
            df = pd.read_csv(file_path)

            # Perform FFT
            new_sample_rate = self.FFT_result(
                file_name = file,
                df = df, 
                dep_var = 'Torque - 1[Nm]',
                length = int(len(df)),
                time_array = df['Time[sec]'], 
                sample_rate=self.config.SAMPLE_RATE,
                is_plot = True,
                )
            new_sample_rate = int(new_sample_rate)
            interval = round(self.config.SAMPLE_RATE / (self.config.MULTI*new_sample_rate))
            df = df[::interval]
  
            # split x, y
            x = df[x_features].values
            y = df[y_features].values
            
            train_idx, valid_idx, test_idx \
                = split_data(n_samples=x.shape[0], valid_ratio=0.1, test_ratio=0.1)
            
            axis = 0
            x, preprocessor_x, y, preprocessor_y \
                = self.transform(x, y, train_idx, axis, ) 

            self.data[file] = {
                "x": x, "y": y, 
                "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx, 
                "preprocessor_x": preprocessor_x, "preprocessor_y": preprocessor_y, 
                }

    def get(self, item=None, dataset=None, return_index=False):
        data = self.data
        if item is None:
            return_index = False
        if dataset is None:
            dataset = data.keys()

        output = []
        if return_index:
            idx_ctr = 0
            idx_dict = {}

        # Find relevent files
        for dataset_ in dataset:
            for file in data.keys(): # compare with the previously added file names
                if file.startswith(dataset_):
                    file_data = data[file]

                    # Output files
                    if item is None:
                        output.append(file)
            
                    # Output items
                    elif item=="train_x":
                        output.append(file_data["x"][file_data["train_idx"], :])
                    elif item=="valid_x":
                        output.append(file_data["x"][file_data["valid_idx"], :])
                    elif item=="test_x":
                        output.append(file_data["x"][file_data["test_idx"], :])
                    elif item=="train_y":
                        output.append(file_data["y"][file_data["train_idx"], :])
                    elif item=="valid_y":
                        output.append(file_data["y"][file_data["valid_idx"], :])
                    elif item=="test_y":
                        output.append(file_data["y"][file_data["test_idx"], :])
                    elif item=="raw_train_x":
                        output.append(file_data["preprocessor_x"].inverse_transform(
                            file_data["x"][file_data["train_idx"], :]
                            ))
                    elif item=="raw_valid_x":
                        output.append(file_data["preprocessor_x"].inverse_transform(
                            file_data["x"][file_data["valid_idx"], :]
                            ))
                    elif item=="raw_test_x":
                        output.append(file_data["preprocessor_x"].inverse_transform(
                            file_data["x"][file_data["test_idx"], :]
                            ))
                    elif item=="raw_x":
                        output.append(file_data["preprocessor_x"].inverse_transform(
                            file_data["x"]
                            ))
                    elif item=="raw_train_y":
                        output.append(file_data["preprocessor_y"].inverse_transform(
                            file_data["y"][file_data["train_idx"], :]
                            ))
                    elif item=="raw_valid_y":
                        output.append(file_data["preprocessor_y"].inverse_transform(
                            file_data["y"][:, file_data["valid_idx"], :]
                            ))
                    elif item=="raw_test_y":
                        output.append(file_data["preprocessor_y"].inverse_transform(
                            file_data["y"][file_data["test_idx"], :]
                            ))
                    elif item=="raw_y":
                        output.append(file_data["preprocessor_y"].inverse_transform(
                            file_data["y"]
                            ))
                    else:
                        output.append(file_data[item])

                    # Save index of each file
                    if return_index:
                        idx_dict[file] = np.arange(idx_ctr, output[-1].shape[1])
                        idx_ctr += output[-1].shape[1]

        if item in [
            "x", "y", 
            "train_x", "valid_x", "test_x", 
            "train_y", "valid_y", "test_y", 
            "raw_train_x", "raw_valid_x", "raw_test_x", "raw_x", 
            "raw_train_y", "raw_valid_y", "raw_test_y", "raw_y", 
            ]:
            output = np.concatenate(output, axis=0)

        if return_index:
            return output, idx_dict
        else:
            return output

