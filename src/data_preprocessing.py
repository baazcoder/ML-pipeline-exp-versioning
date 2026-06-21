import os
import logging
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger("data_preprocessing")
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


class DataPreprocessing:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def preprocess_data(self) -> pd.DataFrame:
        try: 
            logger.debug("Starting text transformation process")
            
            # Convert text to lowercase
            self.df['text'] = self.df['text'].str.lower()

            # Remove duplicate rows
            self.df = self.df.drop_duplicates(keep='first')
            logger.debug("Removed duplicate rows from the DataFrame")
            
            # Tokenize the text
            self.df['text'] = self.df['text'].apply(lambda x: nltk.word_tokenize(x))
            
            # Remove punctuation
            self.df['text'] = self.df['text'].apply(lambda tokens: [word for word in tokens if word not in string.punctuation])

            # Remove stop words
            stop_words = set(stopwords.words('english'))
            self.df['text'] = self.df['text'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])
            logger.debug("Text transformation process completed successfully")

            # removing alphanumeric characters
            self.df['text'] = self.df['text'].apply(lambda tokens: [word for word in tokens if word.isalpha()])
            logger.debug("Removed alphanumeric characters from text")

            # Apply stemming
            porter_stemmer = PorterStemmer()
            self.df['text'] = self.df['text'].apply(lambda tokens: [porter_stemmer.stem(word) for word in tokens])
            logger.debug("Applied stemming to the text")

            
            logger.debug("Text transformation completed")
            return self.df
        except Exception as e:
            logger.error(f"Error in transforming text: {e}")
            raise e
        

def main():
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info("Data loaded successfully for preprocessing")

        train= DataPreprocessing(train_data)
        train_processed = train.preprocess_data()
        test= DataPreprocessing(test_data)
        test_processed = test.preprocess_data()
        logger.info("Data preprocessing completed successfully")

        # Save the preprocessed data
        data_path = os.path.join('./data', 'processed')
        os.makedirs(data_path, exist_ok=True)
        train_processed.to_csv(os.path.join(data_path, 'train_data.csv'), index=False)
        test_processed.to_csv(os.path.join(data_path, 'test_data.csv'), index=False)

        logger.debug("Preprocessed data saved successfully")

    except FileNotFoundError as fnfe:
        logger.error(f"File not found: {fnfe}")
        raise fnfe
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise e

if __name__ == "__main__":
    main()