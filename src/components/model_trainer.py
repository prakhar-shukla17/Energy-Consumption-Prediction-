import os
import sys
import numpy as np
from dataclasses import dataclass
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Initializing XGBRegressor model")
            model = XGBRegressor(
                base_score=0.5,
                booster='gbtree',
                n_estimators=1000,
                early_stopping_rounds=50,
                objective='reg:linear',
                max_depth=3,
                learning_rate=0.01
            )

            logging.info("Fitting the model with early stopping")
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info("Model training complete. Calculating RMSE score.")
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            return rmse

        except Exception as e:
            raise CustomException(e, sys)
1