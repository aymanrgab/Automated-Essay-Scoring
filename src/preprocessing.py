import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess(data):
    stop_words = set(stopwords.words('english'))
    
    def process_text(text):
        if pd.isna(text):
            return ""
        tokens = word_tokenize(str(text).lower())
        return ' '.join([w for w in tokens if w not in stop_words])
    
    data['processed_text'] = data['full_text'].apply(process_text)
    return data