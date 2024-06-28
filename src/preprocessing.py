import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess(data):
    stop_words = set(stopwords.words('english'))
    sia = SentimentIntensityAnalyzer()
    
    def process_text(text):
        if pd.isna(text):
            return ""
        tokens = word_tokenize(str(text).lower())
        return ' '.join([w for w in tokens if w not in stop_words])
    
    def extract_pos_tags(text):
        doc = nlp(text)
        return ' '.join([token.pos_ for token in doc])
    
    def extract_named_entities(text):
        doc = nlp(text)
        return ' '.join([ent.label_ for ent in doc.ents])
    
    def extract_sentiment(text):
        return sia.polarity_scores(text)['compound']
    
    data['processed_text'] = data['full_text'].apply(process_text)
    data['pos_tags'] = data['full_text'].apply(extract_pos_tags)
    data['named_entities'] = data['full_text'].apply(extract_named_entities)
    data['sentiment'] = data['full_text'].apply(extract_sentiment)
    
    return data