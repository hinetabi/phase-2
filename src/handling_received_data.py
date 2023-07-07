import argparse
from problem_config import create_prob_config
import os
import pandas as pd

class HandlingReceivedData:
    def __init__(self) -> None:
        prob_config = {}
        for prob in ["prob-1", "prob-2"]:
            prob_config[prob] = create_prob_config(
                "phase-1", prob_id=prob
            )
            df = pd.DataFrame()
            for dirname, _, filenames in os.walk(prob_config[prob].captured_data_dir):
                for filename in filenames:
                    if filename.endswith(".parquet"):
                        df1 = pd.read_parquet(os.path.join(dirname, filename), engine="fastparquet")
                        df = pd.concat([df, df1], axis=0)
                        print("loaded file name" + os.path.join(dirname, filename))
            df.to_csv(os.path.join(prob_config[prob].captured_data_dir, "processed/all.csv"))
    def preprocess():
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config-path", type=str, required=True)
    # args = parser.parse_args()
    
    handleData =  HandlingReceivedData()
    