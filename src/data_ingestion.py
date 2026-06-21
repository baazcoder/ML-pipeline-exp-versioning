import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


logger = logging.getLogger("data_ingestion")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class DataIngestion:
    def __init__(self, file_path,output_path):
        self.file_path = file_path
        self.output_path = output_path

    def ingest_data(self):
        logger.info("Starting data ingestion process")
        self.load_data()

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.file_path}")
        try:
            data = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully with shape {data.shape}")
            self.preprocess_data(data)
        except pd.errors.ParserError as pe:
            logger.error(f"Parsing error: {pe}")
            raise pe

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by handling missing values and encoding categorical variables."""
        try:
            data.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
            data.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
            data['target'] = data['target'].map({'ham': 0, 'spam': 1})
            logger.info("Data preprocessing completed successfully")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            self.save_data(train_data, test_data)
        except KeyError as ke:
            logger.error(f"Key error during preprocessing: {ke}")
            raise ke
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise e
    
    def save_data(self, train_data: pd.DataFrame,test_data: pd.DataFrame):
        """Save the preprocessed data to a CSV file."""
        try:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            raw_data_path = os.path.join(self.output_path, "raw")
            if not os.path.exists(raw_data_path):
                os.makedirs(raw_data_path)
            train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
            test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
            logger.info(f"Data saved successfully at {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise 

def main():
    file_path = "https://raw.githubusercontent.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/refs/heads/main/experiments/spam.csv"
    output_path = './data'
    data_ingestion = DataIngestion(file_path, output_path)
    data_ingestion.ingest_data()

if __name__ == "__main__":
    main()