import os
from pathlib import Path
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

def merge_and_preprocess_data(file_paths, merge_column, y_column, columns_to_drop):
    """
    Reads and merges data from one or multiple files, and performs data preprocessing.

    Args:
        file_paths (str or list): File path or list of file paths for data files.
        merge_column (str): Name of the column to be used as the merge key.
        y_column (str): Name of the column to be used as the prediction target.
        columns_to_drop (list): List of column names to be dropped.

    Returns:
        df_scaled (pd.DataFrame): Scaled DataFrame after cleaning.
        df_classes (pd.Series): Series containing the prediction target values.
        save_name (str): Name of the saved output file.

    """
    # Create an empty DataFrame to store merged data
    merged_df = pd.DataFrame()

    # Ensure that file_paths is a list even if a single file path is provided
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    # Process and merge data
    for file_path in file_paths:
        print(
            '\n*****************************************************************')
        print("\n-----------------------",
            f"\nSTARTING TO LOAD DATA FROM {file_path}!",
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

        # Merge data using the specified merge_column as the index
        if merged_df.empty:
            merged_df = df
        else:
            # Merge data using the specified merge_column as the index
            merged_df = pd.merge(merged_df, df, on=merge_column,
                                 how='outer')

    # Print the DataFrame
    print(
        '\n*****************************************************************')
    print("\n------------------------",
          "\nSTARTING DATA CLEANING!",
          "\n-------------------------")

    merged_df = merged_df.replace(['?', 'NA', 'na', 'n/a', '', '.'], np.nan)
    merged_df = merged_df.set_index(merge_column)

    # Remove duplicate rows
    print('\n--- Removing duplicate rows ---')
    print("\nShape of df before dups rows meaned:", merged_df.shape)
    merged_df = merged_df.groupby(merged_df.index).mean()
    print("\nShape of df after dups rows meaned:", merged_df.shape)

    # Get predicting column (y)
    df_classes = merged_df[y_column]

    print('\n--- Dropping unwanted columns ---')
    # Drop unwanted columns
    merged_df = merged_df.drop(columns=columns_to_drop)

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
    merged_df = merged_df.loc[:, merged_df.columns != y_column]

    # Remove NAs with too much data missing
    print('\n--- Dropping/imputing columns with too many NAs ---')
    list_col_drop = []
    for i in merged_df.columns:
        count = merged_df[i].isna().sum()
        if count >= 20:
            list_col_drop.append(i)
            merged_df = merged_df.loc[:, merged_df.columns != i]
        else:
            pass

    print("\nTotal number of columns dropped:", len(list_col_drop))

    # Scale the data
    print('\n--- Starting Scaling Data... ---')
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(merged_df), columns=merged_df.columns,
                             index=merged_df.index)
    print('\n--- Completed Scaling! ---')

    # Add class column back in and save
    df_final = pd.concat([pd.Series(df_classes, name=y_column), df_scaled],
                         axis=1)
    print('\n--- Snapshot of final imputed data ---\n',
          df_final.iloc[:5, :5])

    # Prepare the file name for saving
    file_name = os.path.basename(file_paths[0])
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
