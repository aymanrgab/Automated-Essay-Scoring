from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test, data):
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")
    
    # Create a DataFrame to analyze errors
    error_analysis = pd.DataFrame({
        'actual': y_test,
        'predicted': predictions,
        'error': predictions - y_test,
        'text': data['full_text']
    })
    
    # Sort the DataFrame by error magnitude
    error_analysis = error_analysis.sort_values(by='error', key=abs, ascending=False)
    
    # Print the top 10 largest errors
    print("\nTop 10 Largest Errors:")
    print(error_analysis.head(10)[['actual', 'predicted', 'error', 'text']])
