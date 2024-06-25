from src import preprocessing, feature_extraction, model, evaluate
import pandas as pd
import scipy.sparse as sp
import subprocess
import sys

def setup():
    print("Setting up the environment...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Environment setup complete.")

def main():
    setup()

    try:
        # Load and preprocess data
        train_data = preprocessing.load_data('data/train.csv')
        test_data = preprocessing.load_data('data/test.csv')
        
        train_processed = preprocessing.preprocess(train_data)
        test_processed = preprocessing.preprocess(test_data)

        # Extract features
        train_tfidf, train_grammar, test_tfidf, test_grammar = feature_extraction.extract_features(train_processed, test_processed)
        
        # Combine features
        train_combined = sp.hstack((train_tfidf, train_grammar.to_numpy().reshape(-1, 1)))
        test_combined = sp.hstack((test_tfidf, test_grammar.to_numpy().reshape(-1, 1)))

        # Train model
        trained_model, X_test, y_test = model.train(train_combined, train_data['score'])

        # Evaluate model
        evaluate.evaluate_model(trained_model, X_test, y_test)

        # Make predictions on test data
        test_predictions = trained_model.predict(test_combined)

        # Create submission file
        submission = pd.DataFrame({
            'essay_id': test_data['essay_id'],
            'score': test_predictions
        })
        
        # Ensure scores are within the 1-6 range
        submission['score'] = submission['score'].clip(1, 6).round().astype(int)
        
        # Save submission file
        submission.to_csv('submission.csv', index=False)
        print("Submission file created: submission.csv")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data files and column names.")

if __name__ == "__main__":
    main()