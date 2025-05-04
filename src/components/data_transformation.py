import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    

    def create_time_features(self, df):
        df['Datetime'] = pd.to_datetime(df['Datetime'])  # Replace 'Datetime' with the actual column name
        df['hour'] = df['Datetime'].dt.hour
        df['dayofweek'] = df['Datetime'].dt.dayofweek
        df['quarter'] = df['Datetime'].dt.quarter
        df['month'] = df['Datetime'].dt.month
        df['year'] = df['Datetime'].dt.year
        df['dayofyear'] = df['Datetime'].dt.dayofyear
        return df

    # def get_data_transformer_object(self):
    #     '''
    #     This function is responsible for data trnasformation
        
    #     '''
    #     try:
    #         numerical_columns = ["writing_score", "reading_score"]
    #         categorical_columns = [
    #             "gender",
    #             "race_ethnicity",
    #             "parental_level_of_education",
    #             "lunch",
    #             "test_preparation_course",
    #         ]

    #         num_pipeline= Pipeline(
    #             steps=[
    #             ("imputer",SimpleImputer(strategy="median")),
    #             ("scaler",StandardScaler())

    #             ]
    #         )

    #         cat_pipeline=Pipeline(

    #             steps=[
    #             ("imputer",SimpleImputer(strategy="most_frequent")),
    #             ("one_hot_encoder",OneHotEncoder()),
    #             ("scaler",StandardScaler(with_mean=False))
    #             ]

    #         )

    #         logging.info(f"Categorical columns: {categorical_columns}")
    #         logging.info(f"Numerical columns: {numerical_columns}")

    #         preprocessor=ColumnTransformer(
    #             [
    #             ("num_pipeline",num_pipeline,numerical_columns),
    #             ("cat_pipelines",cat_pipeline,categorical_columns)

    #             ]


    #         )

    #         return preprocessor
        
    #     except Exception as e:
    #         raise CustomException(e,sys)

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
            
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")


            logging.info("Creating time features for train and test data.")
            train_df = self.create_time_features(train_df)
            test_df = self.create_time_features(test_df)

            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "PJME_MW"
            feature_columns = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']

            input_feature_train_df = train_df[feature_columns]
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[feature_columns]
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
