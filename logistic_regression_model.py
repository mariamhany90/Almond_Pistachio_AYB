#!/usr/bin/python

import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

class LogisticRegressionModel:
    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val, model_path='./models/logistic_regression_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

    def perform_grid_search(self):
        if not os.path.isfile(self.model_path):
            """Perform grid search to find the best hyperparameters for Logistic Regression."""
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 20, 50, 100],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter': [2000, 3000, 4000]
            }

            logistic_regression = LogisticRegression()
            grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid,
                                       scoring='accuracy', cv=5, n_jobs=-1)
            
            grid_search.fit(self.X_train, self.y_train)
            self.model = grid_search.best_estimator_
            print("Best hyperparameters found:", grid_search.best_params_)
        else:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")

    def evaluate_model(self):
        """Evaluate the model on the test and validation sets."""
        if self.model is None:
            print("Model is not trained yet!")
            return
        
        y_test_pred = self.model.predict(self.X_test)
        y_val_pred = self.model.predict(self.X_val)
        
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_conf_matrix = confusion_matrix(self.y_test, y_test_pred)
        test_roc_auc = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        
        val_accuracy = accuracy_score(self.y_val, y_val_pred)
        val_conf_matrix = confusion_matrix(self.y_val, y_val_pred)
        val_roc_auc = roc_auc_score(self.y_val, self.model.predict_proba(self.X_val)[:, 1])
        
        print(f"Test Accuracy: {test_accuracy:.2f}")
        print("Test Confusion Matrix:\n", test_conf_matrix)
        print(f"Test ROC AUC: {test_roc_auc:.2f}")

        print(f"Validation Accuracy: {val_accuracy:.2f}")
        print("Validation Confusion Matrix:\n", val_conf_matrix)
        print(f"Validation ROC AUC: {val_roc_auc:.2f}")

        return y_test_pred, y_val_pred

    def save_model(self):
        """Save the trained model to a file."""
        if self.model is None:
            print("No model to save!")
            return
            
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def main(self):
        """Main function to orchestrate the loading, training, evaluating, and saving of the model."""
        self.perform_grid_search()
        y_test_pred, y_val_pred = self.evaluate_model()
        self.save_model()
        
        return y_test_pred, y_val_pred