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
        train_features, test_features = feature_extraction.extract_features(train_processed, test_processed)

        # Train model
        trained_model, X_test, y_test = model.train(train_features, train_data['score'])

        # Evaluate model
        evaluate.evaluate_model(trained_model, X_test, y_test, train_processed)

        # Make predictions on test data
        test_predictions = trained_model.predict(test_features)

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