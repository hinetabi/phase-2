import argparse
import logging
from sklearn.svm import SVC
import mlflow
import numpy as np
import xgboost as xgb
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score

from problem_config import (
    ProblemConfig,
    ProblemConst,
    get_prob_config,
)
from raw_data_processor1 import RawDataProcessor
from utils import AppConfig
from catboost import CatBoostClassifier

class ModelTrainer:
    @staticmethod
    def train_model(
        prob_config: ProblemConfig,
        model_params,
        add_captured_data=False,
        experiment_name="xgb-1",
    ):
        logging.info("start train_model")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(
            f"{prob_config.phase_id}_{prob_config.prob_id}_{experiment_name}"
        )

        # load train data
        train_x, train_y = RawDataProcessor.load_train_data(
            prob_config
        )  # Pre-Processing Data
        train_x = train_x.to_numpy()
        train_y = train_y.to_numpy()
        logging.info(f"loaded {len(train_x)} samples")

        # Add Captured
        if add_captured_data:
            captured_x, captured_y = RawDataProcessor.load_capture_data(prob_config)
            captured_x = captured_x.to_numpy()
            captured_y = captured_y.to_numpy()
            train_x = np.concatenate((train_x, captured_x))
            train_y = np.concatenate((train_y, captured_y))
            logging.info(f"added {len(captured_x)} captured samples")

        # train model
        if (
            len(np.unique(train_y)) == 2
        ):  # Đếm số giá trị khác nhau của label để ứng dụng bài toán
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"

        model = CatBoostClassifier(
            learning_rate=0.1,
            random_seed=0,
            depth=8,
            l2_leaf_reg=3.0,
            border_count=254,
        )
        model.fit(train_x, train_y)

        # evaluate
        test_x, test_y = RawDataProcessor.load_test_data(prob_config)
        predictions = model.predict(test_x)
        auc_score = roc_auc_score(test_y, predictions)
        metrics = {"test_auc": auc_score}
        logging.info(f"metrics: {metrics}")
        print(metrics)

        # mlflow log
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = infer_signature(test_x, predictions)
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path=AppConfig.MLFLOW_MODEL_PREFIX + prob_config.prob_id,
            signature=signature,
        )

        mlflow.end_run()
        logging.info("finish train_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--experiment-name", type=str, default=ProblemConst.EXPERIMENT1
    )

    parser.add_argument(
        "--add-captured-data", type=lambda x: (str(x).lower() == "true"), default=False
    )
    args = parser.parse_args()
    args.phase_id = "phase-1"
    args.prob_id = "prob-1"

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    model_config = {"random_state": prob_config.random_state}
    experiment_name = args.experiment_name

    ModelTrainer.train_model(
        prob_config,
        model_config,
        add_captured_data=args.add_captured_data,
        experiment_name=experiment_name,
    )
