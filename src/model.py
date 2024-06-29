from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import os

def train(features, target, model_name='best_model.joblib'):
    model_dir = '../models/'  # Specify the directory to save models
    model_path = os.path.join(model_dir, model_name)  # Construct the full path

    # Create the models directory if it does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory {model_dir} for saving models.")

    # Check if the model already exists
    if os.path.exists(model_path):
        print("Loading the best model from file...")
        best_model = joblib.load(model_path)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        models = {
            'random_forest': {
                'model': RandomForestRegressor(),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 5],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 5],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'support_vector': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1],
                    'kernel': ['linear'],
                    'gamma': ['scale']
                }
            }
        }

        best_model = None
        best_score = 0

        for name, model_info in models.items():
            print(f"Training {name} model...")
            grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='r2', n_jobs=-1, n_iter=10)
            grid_search.fit(X_train, y_train)

            if grid_search.best_score_ > best_score:
                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_

        print(f"Best model: {best_model}")
        print(f"Best R2 score: {best_score}")

        # Save the best model
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

    return best_model, X_test, y_test