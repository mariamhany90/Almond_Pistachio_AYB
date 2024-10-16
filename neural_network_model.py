#!/usr/bin/python

import pandas as pd
import joblib
import os
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Dropout
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

class NeuralNetworkModel:
    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val, model_path='./models/neural_network_model.keras'):
        self.model_path = model_path
        self.model = None
        self.X_train = X_train
        self.y_train = np.where(y_train == 'Existing Customer', 1, 0)
        self.X_test = X_test
        self.y_test = np.where(y_test == 'Existing Customer', 1, 0)
        self.X_val = X_val
        self.y_val = np.where(y_val == 'Existing Customer', 1, 0)

    def create_model(self, num_units=[32], activation='relu', optimizer='adam', dropout_rate=0.5, kernel_regularizer=None):
        model = Sequential()
        model.add(Input(shape=(32,)))  # Correctly specifying as a tuple
        for i in range(len(num_units)):  # Changed range to include the first layer
            model.add(Dense(num_units[i], activation=activation, kernel_regularizer=kernel_regularizer))
            if dropout_rate:  # Add dropout layer
                model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def perform_grid_search(self):
        if not os.path.isfile(self.model_path):
            """Perform grid search to find the best hyperparameters for the neural network."""
            param_grid = {
                'num_units': [[64], [64, 32], [64, 128, 128, 64]],
                'activation': ['relu', 'tanh'],
                'batch_size': [16, 32],
                'epochs': [50, 100],
                'optimizer': ['adam', 'sgd'],
                'dropout_rate': [0.2, 0.5],  # Added dropout rates to grid search
                'kernel_regularizer': [None, keras.regularizers.l2(0.01)]  # L2 regularization options
            }

            best_model = None
            best_score = 0
            best_params = {}

            # Generate all combinations of parameters
            for params in ParameterGrid(param_grid):
                model = self.create_model(
                    num_units=params['num_units'],
                    activation=params['activation'],
                    optimizer=params['optimizer'],
                    dropout_rate=params['dropout_rate'],
                    kernel_regularizer=params['kernel_regularizer']
                )

                # Early stopping callback
                early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                # Fit the model and save the history
                history = model.fit(
                    self.X_train, self.y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_data=(self.X_val, self.y_val),
                    callbacks=[early_stopping],  # Adding early stopping
                    verbose=0
                )

                # Evaluate on validation set
                val_loss, val_accuracy = model.evaluate(self.X_val, self.y_val, verbose=0)
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_model = model
                    best_params = params

            self.model = best_model
            print("Best hyperparameters found:", best_params)
            self.plot_learning_curves(history)  # Plotting learning curves
        else:
            self.model = load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")

    def plot_learning_curves(self, history):
        """Plot learning curves for training and validation loss and accuracy."""
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='best')

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='best')

        plt.tight_layout()
        plt.show()

    def evaluate_model(self):
        """Evaluate the model on the test and validation sets."""
        if self.model is None:
            print("Model is not trained yet!")
            return

        y_test_pred_acc = (self.model.predict(self.X_test) > 0.5).astype("int32")
        y_test_pred = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred_acc)
        test_conf_matrix = confusion_matrix(self.y_test, y_test_pred_acc)
        test_roc_auc = roc_auc_score(self.y_test, y_test_pred)

        print(f"Test Accuracy: {test_accuracy:.2f}")
        print("Test Confusion Matrix:\n", test_conf_matrix)
        print(f"Test ROC AUC: {test_roc_auc:.2f}")

        return y_test_pred

    def save_model(self):
        """Save the trained model to a file."""
        if self.model is None:
            print("No model to save!")
            return

        self.model.save(self.model_path, save_format='keras')  # Save as .keras format
        print(f"Model saved to {self.model_path}")

    def main(self):
        """Main function to orchestrate the loading, training, evaluating, and saving of the model."""
        self.perform_grid_search()
        y_test_pred = self.evaluate_model()
        self.save_model()

        return y_test_pred