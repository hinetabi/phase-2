import argparse
import logging
import os
import random
import time

import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object
from pydantic import BaseModel

from problem_config import ProblemConst, create_prob_config
from raw_data_processor1 import RawDataProcessor as RawDataProcessor1
from raw_data_processor2 import RawDataProcessor as RawDataProcessor2
from utils import AppConfig, AppPath

from scipy import stats
import pyarrow.parquet as pq

PREDICTOR_API_PORT = 8000


class Data(BaseModel):
    id: str
    rows: list
    columns: list


class ModelPredictor:
    def __init__(self, config_file_path):

        config_file_path_specific = {
            "prob-1": "prob-1/model-1.yaml",
            "prob-2": "prob-2/model-1.yaml",
        }

        self.model = {}
        self.config = {}
        self.category_index = {}
        self.prob_config = {}
        self.pFile = {}
        self.train_features = {}
        self.max_feq_data = {}
        for prob in ["prob-1", "prob-2"]:
            with open(
                os.path.join(config_file_path, config_file_path_specific[prob]), "r"
            ) as f:
                self.config[prob] = yaml.safe_load(f)
            logging.info(f"model-config: {self.config[prob]}")

            mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

            self.prob_config[prob] = create_prob_config(
                self.config[prob]["phase_id"], self.config[prob]["prob_id"]
            )

            # load category_index

            if prob == 'prob-1':
                self.category_index[prob] = RawDataProcessor1.load_category_index(self.prob_config[prob])
                self.max_feq_data[prob] = RawDataProcessor1.load_max_feq_dict(self.prob_config[prob])
            else:
                self.category_index[prob] = RawDataProcessor2.load_category_index(self.prob_config[prob])
                self.max_feq_data[prob] = RawDataProcessor2.load_max_feq_dict(self.prob_config[prob])


            # load model
            model_uri = os.path.join(
                "models:/",
                self.config[prob]["model_name"],
                str(self.config[prob]["model_version"]),
            )
            model_uri = model_uri.replace("\\", "/")
            self.model[prob] = mlflow.pyfunc.load_model(model_uri)

            # load data drift
            self.pFile[prob] = pq.ParquetFile(self.prob_config[prob].train_x_path)
            self.train_features[prob] = self.pFile[prob].read().to_pandas()


        
    def detect_drift(self, feature_df, prob) -> int:

        num_features = self.train_features[prob].shape[1]
        significance_level = 0.05

        for i in range(num_features):
            train_data = self.train_features[prob][f'feature{i+1}']
            test_data = feature_df[f'feature{i+1}']

            _, p_value = stats.ks_2samp(train_data, test_data)

            if p_value > significance_level:
                pass
            else:
                return 0
        return 1


    def predict(self, data: Data, prob="prob-1"):
        start_time = time.time()

        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)


        # save request data for improving models
        ModelPredictor.save_request_data(
            raw_df, self.prob_config[prob].captured_data_dir, data.id
        )
        
        if prob == "prob-1":
            # handling missing data, replace missing with self.max_feq_data
            for column in raw_df.columns:
                try:
                    self.max_feq_data[prob][column] = float(self.max_feq_data[prob][column])
                except:
                    raw_df[column] = raw_df[column].fillna(self.max_feq_data[prob][column])
            # logging.info(f"missing data replaced with self.max_feq_data")
            # logging.info(f"missing data count = {raw_df.isna().sum()}")
            feature_df = RawDataProcessor1.apply_category_features(
                raw_df=raw_df,
                categorical_cols=self.prob_config[prob].categorical_cols,
                category_index=self.category_index[prob],
            )
        else:
            for column in raw_df.columns:
                try:
                    self.max_feq_data[prob][column] = float(self.max_feq_data[prob][column])
                except:
                    raw_df[column] = raw_df[column].fillna(self.max_feq_data[prob][column])

            feature_df = RawDataProcessor2.apply_category_features(
                raw_df=raw_df,
                categorical_cols=self.prob_config[prob].categorical_cols,
                category_index=self.category_index[prob],
            )

        # logging.info(f'feature df = {feature_df}')
        feature_df = feature_df.astype("float")
        prediction = self.model[prob].predict(feature_df)
        is_drifted = self.detect_drift(feature_df, prob)

        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction takes {run_time} ms")
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        output_file_path = os.path.join(captured_data_dir, f"{filename}.csv")
        feature_df.to_csv(output_file_path, index=False)
        return output_file_path


class PredictorApi:
    def __init__(self, predictor: ModelPredictor):
        self.predictor = predictor
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/phase-1/prob-1/predict")
        async def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor.predict(data, prob="prob-1")
            self._log_response(response)
            return response

        @self.app.post("/phase-1/prob-2/predict")
        async def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor.predict(data, prob="prob-2")
            self._log_response(response)
            return response

    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    # default_config_path = (
    #     AppPath.MODEL_CONFIG_DIR
    #     / ProblemConst.PHASE1
    #     / ProblemConst.PROB1
    #     / "model-1.yaml"
    # ).as_posix()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--port", type=int, default=5040)
    args = parser.parse_args()

    # args.config_path = "data/model_config/phase-1/"
    predictor = ModelPredictor(config_file_path=args.config_path)


    api = PredictorApi(predictor)
    api.run(port=args.port)
