#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataPipeline:
    def __init__(self, data):
        self.data = data

    def drop_duplicate_id(self, column):
        """Drop duplicates based on a specific column."""
        self.data = self.data.drop_duplicates(subset=[column], keep='first')
        return self

    def calculate_age(self, date_of_birth_column, age_column):
        """Calculate 'Age' from 'Date_of_birth' and drop the 'Date_of_birth' column."""
        current_year = pd.Timestamp.now().year
        self.data[age_column] = current_year - pd.to_datetime(self.data[date_of_birth_column]).dt.year
        self.data = self.data.drop(columns=[date_of_birth_column])
        return self

    def change_dtype_to_string(self):
        """Convert object data types to string data types."""
        self.data = self.data.astype({col: 'string' for col in self.data.select_dtypes(include='object').columns})
        return self

    def sort_and_drop_duplicates(self, sort_columns, na_position='last'):
        """Sort data, move NaN to the end, drop duplicates, and resort."""
        self.data = self.data.sort_values(by=sort_columns,ascending=True, na_position=na_position)
        self.data = self.data.drop_duplicates(subset=['Attrition_Flag', 'Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'], keep='first')
        self.data = self.data.sort_values(by=sort_columns[0]) 
        return self

    def fill_na_with_unknown(self):
        """Fill NaN values with 'Unknown'."""
        self.data = self.data.fillna('Unknown')
        return self

    def drop_first_column(self):
        """Drop the first column in the DataFrame."""
        self.data = self.data.iloc[:, 1:]
        return self

    def drop_high_correlation_columns(self, threshold=0.8):
        """Drop one of two columns with correlation higher than the threshold."""
        numeric_columns = self.data.select_dtypes(include=['float64', 'int64', 'int32']).columns
        corr_matrix = self.data[numeric_columns].corr()
        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap', fontsize=16)
        plt.savefig('./figures/correlation_heatmap.jpg', format='jpg', dpi=300)
        plt.close()

        corr_matrix = self.data[numeric_columns].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = []

        for col in upper_triangle.columns:
            high_corr_cols = upper_triangle[col][upper_triangle[col] > threshold].index.tolist()
        
            for corr_col in high_corr_cols:
                if corr_col not in to_drop:
                    unique_col = self.data[col].nunique()
                    unique_corr_col = self.data[corr_col].nunique()

                    if unique_col > unique_corr_col:
                        to_drop.append(corr_col)
                    else:
                        to_drop.append(col)

        self.data = self.data.drop(columns=list(set(to_drop))) 
        numeric_columns = self.data.select_dtypes(include=['float64', 'int64', 'int32']).columns
        corr_matrix = self.data[numeric_columns].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap After Removing High R', fontsize=16)
        plt.savefig('./figures/correlation_heatmap_after.jpg', format='jpg', dpi=300)
        plt.close()
        
        return self
