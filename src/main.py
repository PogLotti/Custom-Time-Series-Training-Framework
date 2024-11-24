from pathlib import Path
from configuration import Configuration
from data_helpers import DataHolder, SINDyDataHolder
from train import train
from predict import predict

def main():

    config = Configuration()

    # ---------------------- Data preprocessing --------------------- #
    data_holder = DataHolder(config)
    data_holder.add()

    train_x = data_holder.get(item="train_x")
    train_y = data_holder.get(item="train_y")
    valid_x = data_holder.get(item="valid_x")
    valid_y = data_holder.get(item="valid_y")
    test_x = data_holder.get(item="test_x")
    test_y = data_holder.get(item="test_y")

    # ---------------------- Model initialization --------------------- #
    config.model = "lstm"
    
    forecast_model = config.init_model(
        input_dim = train_x.shape[1],
        output_dim = train_y.shape[1],
    )

if __name__ == '__main__':
    main()
