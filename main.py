#!/usr/bin/python

import pandas as pd
import numpy as np
from data_pipeline import DataPipeline
from data_preprocess import DataPreprocessor
from logistic_regression_model import LogisticRegressionModel
from random_forest_classifier_model import RandomForestModel
from neural_network_model import NeuralNetworkModel
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
import joblib
from model_evaluation import evaluate_model
import os

def main():
    cleaned_data_file = './data/cleaned_data.csv'

    if os.path.isfile(cleaned_data_file):
        cleaned_data = pd.read_csv(cleaned_data_file)
        print("Loaded cleaned data from 'cleaned_data.csv'.")
    else:
        data_path = './data/credit-card_customers.xlsx'
        data = pd.read_excel(data_path)
        
        pipeline = DataPipeline(data)

        cleaned_data = (pipeline
            .drop_duplicate_id(column='Customer_Number')
            .calculate_age(date_of_birth_column='Date_of_birth', age_column='Age')
            .change_dtype_to_string()
            .sort_and_drop_duplicates(sort_columns=['Age', 'Dependent_count', 'Marital_Status', 'Income_Category'])
            .fill_na_with_unknown()
            .drop_first_column()
            .drop_high_correlation_columns(threshold=0.8)
            .data)

        cleaned_data.to_csv('./data/cleaned_data.csv', index=False)

    cleaned_data = cleaned_data.astype({col: 'string' for col in cleaned_data.select_dtypes(include='object').columns})
    cleaned_data = cleaned_data.astype({col: 'int64' for col in cleaned_data.select_dtypes(include='int32').columns})

    if not os.path.isfile('./data/train/X_train.csv'):
        preprocessor = DataPreprocessor(cleaned_data)
        X_train, X_val, X_test, X_unknown, y_train, y_val, y_test = preprocessor.preprocess(target_column='Attrition_Flag')
        X_train.to_csv('./data/train/X_train.csv', index=False)
        y_train.to_csv('./data/train/y_train.csv', index=False)
        X_test.to_csv('./data/test/X_test.csv', index=False)
        X_val.to_csv('./data/test/X_val.csv', index=False)
        y_test.to_csv('./data/test/y_test.csv', index=False)
        y_val.to_csv('./data/test/y_val.csv', index=False)
        X_unknown.to_csv('./data/unknown/X_unknown.csv', index=False)
    
    try:
        X_train
    except NameError:
        X_train = pd.read_csv('./data/train/X_train.csv')
        y_train = pd.read_csv('./data/train/y_train.csv').values.ravel()
        X_test = pd.read_csv('./data/test/X_test.csv')
        X_val = pd.read_csv('./data/test/X_val.csv')
        y_test = pd.read_csv('./data/test/y_test.csv').values.ravel()
        y_val = pd.read_csv('./data/test/y_val.csv').values.ravel()
        X_unknown = pd.read_csv('./data/unknown/X_unknown.csv')

    LR_model = LogisticRegressionModel(X_train, y_train, X_test, y_test, X_val, y_val)
    y_test_LR_pred, y_val_LR_pred = LR_model.main()

    RFC_model = RandomForestModel(X_train, y_train, X_test, y_test, X_val, y_val)
    y_test_RFC_pred, y_val_RFC_pred = RFC_model.main()

    NN_model = NeuralNetworkModel(X_train, y_train, X_test, y_test, X_val, y_val)
    y_test_NN_pred = NN_model.main()

    print(y_test_LR_pred)
    print(y_test_NN_pred)
    print(y_test_RFC_pred)

    nn_model = load_model('./models/neural_network_model.keras')
    rf_model = joblib.load('./models/random_forest_model.joblib')
    lr_model = joblib.load('./models/logistic_regression_model.joblib')

    evaluate_model(X_test, y_test, lr_model, 'Logistic Regression')
    evaluate_model(X_test, y_test, rf_model, 'Random Forest')
    evaluate_model(X_test, y_test, nn_model, 'Neural Network')


if __name__ == '__main__':
    main()
