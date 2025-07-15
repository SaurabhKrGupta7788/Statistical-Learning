
import pandas as pd
import numpy as np


# Function to load data from a CSV file
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None

# Function to remove duplicate rows
def drop_duplicates(df):
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print("Duplicates removed:", before - after)
    return df

# Function to handle missing values
def handle_missing(df, method='drop'):
    if method == 'drop':
        df = df.dropna()
        print("Missing values dropped.")
    elif method == 'fill_zero':
        df = df.fillna(0)
        print("Missing values filled with 0.")
    elif method == 'fill_mean':
        df = df.fillna(df.mean(numeric_only=True))
        print("Missing values filled with mean.")
    else:
        print("Invalid method. No missing value handling applied.")
    return df

# Function to save the cleaned data
def save_cleaned_data(df, output_path='cleaned_data.csv'):
    df.to_csv(output_path, index=False)
    print("Cleaned data saved to", output_path)

# Main cleaning pipeline function
def clean_data(file_path, save_path='cleaned_data.csv', method='drop'):
    df = load_data(file_path)
    if df is not None:
        df = drop_duplicates(df)
        df = handle_missing(df, method=method)
        save_cleaned_data(df, save_path)
        print(df.isna().sum())  # just after clean_data()

        return df