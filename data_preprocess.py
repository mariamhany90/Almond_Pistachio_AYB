#!/usr/bin/python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self, target_column, test_size=0.4, val_size=0.5):
        X_unknown = self.data[self.data[target_column] == 'Unknown'].copy()
        X_unknown.drop(columns=[target_column], inplace=True)
        known_data = self.data[self.data[target_column] != 'Unknown']

        X_train, X_temp, y_train, y_temp = train_test_split(
            known_data.drop(columns=[target_column]),
            known_data[target_column],
            test_size=test_size,
            stratify=known_data[target_column],
            random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=42
        )

        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['string']).columns.tolist()
        
        X_train_dummies = pd.get_dummies(X_train[categorical_cols], drop_first=False)
        X_val_dummies = pd.get_dummies(X_val[categorical_cols], drop_first=False)
        X_test_dummies = pd.get_dummies(X_test[categorical_cols], drop_first=False)
        X_unknown_dummies = pd.get_dummies(X_unknown[categorical_cols], drop_first=False)

        X_train_dummies = X_train_dummies.loc[:, ~X_train_dummies.columns.str.contains('Unknown')]
        X_val_dummies = X_val_dummies.loc[:, ~X_val_dummies.columns.str.contains('Unknown')]
        X_test_dummies = X_test_dummies.loc[:, ~X_test_dummies.columns.str.contains('Unknown')]
        X_unknown_dummies = X_unknown_dummies.loc[:, ~X_unknown_dummies.columns.str.contains('Unknown')]

        scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
        X_val_scaled = scaler.transform(X_val[numeric_cols])
        X_test_scaled = scaler.transform(X_test[numeric_cols])
        X_unknown_scaled = scaler.transform(X_unknown[numeric_cols])

        X_train_final = pd.concat([pd.DataFrame(X_train_scaled, columns=numeric_cols), X_train_dummies.reset_index(drop=True)], axis=1)
        X_val_final = pd.concat([pd.DataFrame(X_val_scaled, columns=numeric_cols), X_val_dummies.reset_index(drop=True)], axis=1)
        X_test_final = pd.concat([pd.DataFrame(X_test_scaled, columns=numeric_cols), X_test_dummies.reset_index(drop=True)], axis=1)

        return X_train_final, X_val_final, X_test_final, X_unknown, y_train, y_val, y_test