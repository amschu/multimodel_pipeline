"""Step 1 of MultiColorMaize pipeline: Data Cleaning"""

import os
from pathlib import Path
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os


def read_data(file_path):
    """
    Reads the data from a CSV file and returns it as a pandas DataFrame.

    Args:
        file_path (str): Path of the CSV file

    Returns:
        df (pd.DataFrame): DataFrame containing the data from the CSV file.
        csvfile_path (Path): Path object of the CSV file.
    """
    print(
        '\n*****************************************************************')
    print("\n-----------------------",
          "\nSTARTING TO LOAD DATA!",
          "\n-----------------------\n")
    # Check if file path exists
    csvfile_path = Path(file_path)
    if not csvfile_path.exists():
        print("Oops, file doesn't exist!")
    else:
        print("Yay, the file exists!")

    # Open the CSV file using the csv module
    with open(csvfile_path, 'r', encoding="utf-8") as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)
        # Read the first row as the header row
        header = next(csv_reader)
        # Read the remaining rows as data
        data = list(csv_reader)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data, columns=header)

    # Print the DataFrame
    print("\n----------------------------",
          "\nDATA SUCCESSFULLY LOADED IN!",
          "\n----------------------------")

    return df, csvfile_path

def clean_data(df, y_column, data_path, columns_to_drop):
    """Cleans the data by performing various data cleaning operations.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        y_column (str): Name of the column to be used as the prediction target.
        data_path (str): Path of the data file.
        columns_to_drop (list): List of column names to be dropped.

    Returns:
        df_scaled (pd.DataFrame): Scaled DataFrame after cleaning.
        df_classes (pd.Series): Series containing the prediction target values.
        save_name (str): Name of the saved output file.

    NOTES: cleaning operations
        - change any na related characters to np.nan
        - remove duplicate rows by grouping by index and meaning row values
        - drop unwanted columns from dataframe
        - encode classes as numeric, handling binary, multi-class, or non-numeric targets
        - remove columns with a total number of nan values > 20
        - scale dataframe
    """
    # Print the DataFrame
    print(
        '\n*****************************************************************')
    print("\n------------------------",
          "\nSTARTING DATA CLEANING!",
          "\n-------------------------")

    df = df.replace(['?', 'NA', 'na', 'n/a', '', '.'], np.nan)
    df = df.set_index('GRIN')

    # Remove duplicate rows
    print('\n--- Removing duplicate rows ---')
    print("\nShape of df before dups rows meaned:", df.shape)
    df = df.groupby(df.index).mean()
    print("\nShape of df after dups rows meaned:", df.shape)

    # Get predicting column (y)
    df_classes = df[y_column]

    print('\n--- Dropping unwanted columns ---')
    # Drop unwanted columns
    df = df.drop(columns=columns_to_drop)

    # Handling different types of y_column
    if pd.api.types.is_numeric_dtype(df_classes):
        # If the y_column is already numeric, no need for further conversion
        pass
    elif pd.api.types.is_bool_dtype(df_classes):
        # If y_column is binary (boolean), convert to 0 or 1
        df_classes = df_classes.astype(int)
    else:
        # If y_column is not numeric, encode it using LabelEncoder
        le = LabelEncoder()
        df_classes = le.fit_transform(df_classes)

    # Print the updated DataFrame
    print("\n--- Converted Y_col to numeric classes ---\n",
          pd.Series(df_classes).value_counts())

    # Drop y_column
    df = df.loc[:, df.columns != y_column]

    # Remove NAs with too much data missing
    print('\n--- Dropping/imputing columns with too many NAs ---')
    list_col_drop = []
    for i in df.columns:
        count = df[i].isna().sum()
        if count >= 20:
            list_col_drop.append(i)
            df = df.loc[:, df.columns != i]
        else:
            pass

    print("\nTotal number of columns dropped:", len(list_col_drop))

    # Scale the data
    print('\n--- Starting Scaling Data... ---')
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns,
                             index=df.index)
    print('\n--- Completed Scaling! ---')

    # Add class column back in and save
    df_final = pd.concat([pd.Series(df_classes, name=y_column), df_scaled],
                         axis=1)
    print('\n--- Snapshot of final imputed data ---\n',
          df_final.iloc[:5, :5])

    # Prepare the file name for saving
    file_name = os.path.basename(data_path)
    save_name = file_name.replace('.csv',
                                  '') + '_predicting_' + y_column + '_preprocessed.txt'

    # Save the final DataFrame to a file
    df_final.to_csv(save_name, header=True)

    print(f"\nOutput file saved as: {save_name}")
    print("\n-----------------------",
          "\nDATA CLEANING COMPLETE!",
          "\n-----------------------")

    # Return the scaled data, class column, and the file name
    return df_scaled, df_classes, save_name
