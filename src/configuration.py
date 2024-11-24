import torch
from LSTM.lstm import LSTM
from pathlib import Path
import json

class Configuration():
    
    def __init__(self):
        
        torch.manual_seed(42)

        # Data
        self.file = ""
        self.dataset_prefix = ""
        self.version = []
        self.features = []
  
        # FFT
        self.TIME_LEN = None
        self.SAMPLE_RATE = None
        self.MULTI = None
        
        # Model
        self.model = None
        self.stopping_criteria = ["validation loss"]
        self.validation_max_epochs = None
        self.stopping_count = None

        # Path
        self.root_dir = Path.cwd().parent
        self.data_path = Path().joinpath(self.root_dir, "data")
        self.output_path = Path().joinpath(self.root_dir, "output")
        self.src_path = Path().joinpath(self.root_dir, "src")
        self.lstm_path = Path().joinpath(self.src_path, "lstm")
        self.fft_path = Path().joinpath(self.output_path, "fft_result")
        
        self.mkdir_all()

    def mkdir_all(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.sindy_path.mkdir(parents=True, exist_ok=True)
        self.pinn_path.mkdir(parents=True, exist_ok=True)
        self.dnn_path.mkdir(parents=True, exist_ok=True)
        self.fft_path.mkdir(parents=True, exist_ok=True)
        self.sindy_output_path.mkdir(parents=True, exist_ok=True)
        self.pinn_output_path.mkdir(parents=True, exist_ok=True)
        self.dnn_output_path.mkdir(parents=True, exist_ok=True)
        if self.model == "lstm":
            self.lstm_path.mkdir(parents=True, exist_ok=True)


    def get_model_output_path(self, model):
        if model == "lstm":
            return self.lstm_path

    def init_model(self, input_dim, output_dim):

        # Collect all hyperparameters in a dictionary
        hyperparameters = {
            "model": self.model,
            "input_dim": input_dim,
            "output_dim": output_dim,
        }

        # Add any common hyperparameters
        common_hyperparameters = {
            "max_epochs": 500,
            "stopping_count": 20,
            "lr": 1e-2,
            "batch_size": 4096,
            "dropout": 0,
            "n_layers": 4,
            "hidden_dim": 32,
        }
        hyperparameters.update(common_hyperparameters)
        
        if self.model == "mlp":

            self.forecast_model = MLP(
                config = self,
                input_dim = hyperparameters["input_dim"],
                n_layers = hyperparameters["n_layers"],
                hidden_dim = hyperparameters["hidden_dim"],
                output_dim = hyperparameters["output_dim"],
                lr = hyperparameters["lr"],
                batch_size = hyperparameters["batch_size"],
                max_epochs = hyperparameters["max_epochs"],
                dropout = hyperparameters["dropout"],
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            )
            
        if self.model == "lstm":    
            
            self.forecast_model = LSTM(
                config = self,
                input_dim = hyperparameters["input_dim"],
                n_layers = hyperparameters["n_layers"],
                hidden_dim = hyperparameters["hidden_dim"],
                output_dim = hyperparameters["output_dim"],
                lr = hyperparameters["lr"],
                batch_size = hyperparameters["batch_size"],
                max_epochs = hyperparameters["max_epochs"],
                dropout = hyperparameters["dropout"],
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            )

        # Get the output path and ensure the directory exists
        output_path = self.get_model_output_path(self.model)
        hyperparameters_file = output_path / 'hyperparameters.json'
        
        # Save the hyperparameters to a JSON file
        try:
            hyperparameters_file.write_text(json.dumps(hyperparameters, indent=1))
            print(f"Hyperparameters saved to {hyperparameters_file}")
        except PermissionError:
            print(f"Warning: cannot write hyperparameters to {hyperparameters_file}")
        except Exception as e:
            print(f"Warning: store hyperparameters error: {str(e)}")

        return self.forecast_model
    
