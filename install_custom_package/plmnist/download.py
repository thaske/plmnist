import argparse

from model import LitMNIST
from config import DATA_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_PATH)
    args = parser.parse_args()

    LitMNIST(data_dir=args.data_dir).prepare_data()
