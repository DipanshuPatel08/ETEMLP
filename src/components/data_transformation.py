import sys
import os 
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class dataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')
    
class dataTransforamation:
    def __init__(self):
        self.transformaton_Config = dataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is reposible for data transformation for various types of data
        '''
        
        try:
            raw_data_path = os.path.join('artifacts', 'raw.csv')
            df = pd.read_csv(raw_data_path)
            logging.info("Raw data loaded for feature identification")
            
            numerical_features = df.select_dtypes(exclude="object").columns.tolist()
            categorical_features = df.select_dtypes(include="object").columns.tolist()
            logging.info("Divided in to Numerical and Categorical Features")
            logging.info(f"numerical_features: {numerical_features}")
            logging.info(f"categorical_features: {categorical_features}")
            
            # Numerical Pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logging.info("Numerical and Categorcial pipelines are successfully created")
            
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numerical_features),
                ('cat', cat_pipeline, categorical_features)
            ])
            logging.info("The coloumn Transformer has been succesfully implemented")
            
            return preprocessor
        

        except Exception as e:
            logging.error("Error in get_data_transformer_object")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read Train and Test data completed')
            
            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_object()
            
            target_coloumn_name = 'math_score'
            numerical_coloumns = ['writing_score', 'reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_coloumn_name],axis=1)
            target_feature_train_df = train_df[target_coloumn_name]
            
            input_feature_test_df = test_df.drop(columns=[target_coloumn_name],axis=1)
            target_feature_test_df = test_df[target_coloumn_name]
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info(f"Saved Preprocessing object")
            
            save_object (
                file_path = self.get_data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        