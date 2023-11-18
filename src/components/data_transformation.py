import sys
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
import os
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformed_object(self):
        try:
            #defining column
            categorical_col=["Gender"]
            numerical_col=["CreditScore","Age","Tenure","Balance","NumOfProducts",
                           "HasCrCard","IsActiveMember","EstimatedSalary"]
            
            # defining gender
            Gender_catagories=["Male","Female","Other"]

            logging.info("pipeline initiated")
            # numerical pipeline 
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            ) 
            # categorical column
            cat_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                       ("ordinalencoder",OrdinalEncoder(categories=[Gender_catagories])),
                       ("scaler",StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_col),
                ("cat_pipeline",cat_pipeline,categorical_col)
            ])

            return preprocessor

            logging.info("pipeline completed")
            
        except Exception as e:
            logging.info("error in data DataTransformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # reading dataset

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info(f'checking shape of data{train_df.shape,test_df.shape}')

            logging.info("read train and test data completes")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info("obtaining preprocessor object")
            preprocessing_obj=self.get_data_transformed_object()

            














































