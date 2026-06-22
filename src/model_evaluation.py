import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live


log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger("model_evaluation")
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, "model_evaluation.log")
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

def load_model(model_path: str):
    """load trained model"""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    logger.debug(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully with shape {data.shape}")
        return data
    
    except pd.errors.ParserError as pe:
        logger.error("failed to parse the csv file", pe)

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e
    
def evaluate_model(clf, x_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """Evaluate model and return the evaluation metrices"""
    try:
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,y_pred)

        metrics_dict = {
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "Roc_Auc": roc_auc

        }

        logger.debug("Model evaluation completed successfully")

        return metrics_dict
    except Exception as e:
        logger.debug("something went wrong while evaluation", e)
        raise

def save_metrices(metrices: dict, file_path: str) -> None:
    """Save the evaluation metrices to json file"""

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrices, f, indent = 4)
        
        logger.debug("Evaluation metrics saved!")
    except Exception as e:
        logger.debug("Error occured while saving",e)
        raise

def main():
    try:
        params = load_params('params.yaml')



        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf,X_test,y_test)

        # Experiment tracking using dvclive
        with Live(save_dvc_exp = True) as live:
            live.log_metric("accuracy: ", accuracy_score(y_test, y_test))
            live.log_metric("precision: ", precision_score(y_test, y_test))
            live.log_metric("recall: ", recall_score(y_test, y_test))

            live.log_params(params)

        save_metrices(metrics,'reports/metrics.json')
    except Exception as e:
        logger("error while evaluating model",e)
        raise

if __name__ == '__main__':
    main()