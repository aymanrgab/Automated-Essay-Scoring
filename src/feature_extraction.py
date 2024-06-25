from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import subprocess
import sys

def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        print("Downloading spaCy model. This may take a while...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

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
    
    return train_tfidf, train_grammar, test_tfidf, test_grammar