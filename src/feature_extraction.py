import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_features(train_data, test_data=None):
    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000)
    
    if test_data is not None:
        # Fit on both train and test data to ensure consistent vocabulary
        all_text = train_data['processed_text'].tolist() + test_data['processed_text'].tolist()
        vectorizer.fit(all_text)
        
        train_tfidf = vectorizer.transform(train_data['processed_text'])
        test_tfidf = vectorizer.transform(test_data['processed_text'])
    else:
        train_tfidf = vectorizer.fit_transform(train_data['processed_text'])
        test_tfidf = None
    
    # Grammar features
    def count_grammar_errors(text):
        doc = nlp(text)
        return len([token for token in doc if token.dep_ == 'ROOT'])
    
    train_grammar = train_data['full_text'].apply(count_grammar_errors)
    
    if test_data is not None:
        test_grammar = test_data['full_text'].apply(count_grammar_errors)
    else:
        test_grammar = None
    
    # POS, NER, and Sentiment features
    train_pos = train_data['pos_tags'].apply(lambda x: len(x.split()))
    train_ner = train_data['named_entities'].apply(lambda x: len(x.split()))
    train_sentiment = train_data['sentiment']
    
    if test_data is not None:
        test_pos = test_data['pos_tags'].apply(lambda x: len(x.split()))
        test_ner = test_data['named_entities'].apply(lambda x: len(x.split()))
        test_sentiment = test_data['sentiment']
    else:
        test_pos = None
        test_ner = None
        test_sentiment = None

    train_combined = sp.hstack((train_tfidf, 
                                np.array(train_grammar).reshape(-1, 1), 
                                np.array(train_pos).reshape(-1, 1), 
                                np.array(train_ner).reshape(-1, 1), 
                                np.array(train_sentiment).reshape(-1, 1)))
    
    if test_data is not None:
        test_combined = sp.hstack((test_tfidf, 
                                   np.array(test_grammar).reshape(-1, 1), 
                                   np.array(test_pos).reshape(-1, 1), 
                                   np.array(test_ner).reshape(-1, 1), 
                                   np.array(test_sentiment).reshape(-1, 1)))
    else:
        test_combined = None

    return train_combined, test_combined