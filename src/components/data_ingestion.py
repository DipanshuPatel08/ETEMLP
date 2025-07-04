import os 
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import dataTransforamation,dataTransformationConfig

@dataclass
class dataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')   
    raw_data_path:str = os.path.join('artifacts','raw.csv')

class dataIngestion:
    def __init__(self):
        self.ingestion_config = dataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('/Users/dipanshu_08/Desktop/Programming/ETEMLP/notebook/data/stud.csv')
            logging.info('Read the database as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("train test split Initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of data is completed')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            
        
if __name__ == "__main__":
    obj = dataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation = dataTransforamation()
    data_transformation.initiate_data_transformation(train_data,test_data)