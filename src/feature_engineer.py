import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml


log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger("feature_engineer")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, "feature_engineer.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(file_path: str):
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Params loaded successfully from yaml file")
        return params
    except FileNotFoundError as fnf:
        logger.debug("Couldn't find the file",fnf)
        raise
    except Exception as e:
        logger.debug("Error occured while loading params from yaml file", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    logger.debug(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        data.fillna('',inplace = True)
        logger.debug(f"Data loaded successfully with shape {data.shape}")
        return data
    
    except pd.errors.ParserError as pe:
        logger.error("failed to parse the csv file", pe)

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int):
    """apply tfidf"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        train_df['target'] = y_train
        test_df['target'] = y_test
        logger.debug("TF-IDF transformation completed successfully")
        
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error in TF-IDF transformation: {e}")
        raise e
    

def save_data(df, file_path):
    """Save DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise e
    
def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineer']['max_features']
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features = max_features)

        save_data(train_df, './data/processed/train_tfidf.csv')
        save_data(test_df, './data/processed/test_tfidf.csv')

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise e
    
if __name__ == "__main__":
    main()