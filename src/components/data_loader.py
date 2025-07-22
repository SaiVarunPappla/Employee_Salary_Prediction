import pandas as pd
import os
from src.logger import logging
from src.exception import CustomException

def load_data(file_path: str) -> pd.DataFrame:
    try:
        logging.info(f"Reading data from {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e, sys)
