import os
import pandas as pd

def get_top_model_name(results_df, score_metric="F1-score"):
    """
    Retrieves the feature names of the top-performing model from the results
    DataFrame based on a specified score metric.

    Args:
        results_df (pandas.DataFrame): DataFrame containing the performance
        scores of different models.

        score_metric (str): Metric to determine the top-performing model
        (default: "F1-score").

    Returns:
        str: Name of the top-performing model.
    """
    print('\n*****************************************************************')
    print("\n---------------------------------------",
          "\nGETTING THE TOP PERFORMING MODELS NAME",
          "\n----------------------------------------")
    # get df
    results_df.reset_index(inplace=True)
    # get column names from results table
    results_cols = results_df.columns

    # get the first column name that houses all metric values
    col_name = str(results_cols[0])
    # print("Column name with metric scores:", col_name)

    # get row index that contains the metric score of interest
    row_index = results_df[results_df[col_name] == score_metric].index[0]

    # print("\nRow index:", row_index)

    # # Print the row index of metric score
    # print(row_index)

    # reset index to column with metric scores
    results_df.set_index(col_name, inplace=True)

    # check that the metric values have been made index
    # print("Checking df index after resetting\n", results_df.head())

    # Get the values in the specified metric row
    row_values = results_df.iloc[row_index].values

    # Get the indices that would sort the row values in ascending order
    sorted_indices = row_values.argsort()

    # print("Column index from the smallest value to largest:", sorted_indices)

    # get the lowest score column index
    low_col_index = sorted_indices[0]

    # get the highest scoring column index
    top_col_index = sorted_indices[-1]

    # print the index columns with the highest and lowest value
    # print(top_col_index, low_col_index)

    # create a separate df with the lowest value
    df_low = results_df.iloc[row_index, low_col_index]

    # create a separate df with the highest value
    df_high = results_df.iloc[row_index, top_col_index]

    # look at the dataframes with the highest and lowest values
    # print("df_high\n", df_high)
    # print("\ndf_low\n", df_low)

    # get the model name from the column to use only the top model
    cols_list = results_df.columns
    top_model_name = cols_list[top_col_index]

    print("The top model is:", top_model_name)
    # confirming that the top results is indeed to the top model name
    # log = results_df[top_model_name]
    # print(log)
    print("\n------------------",
          "\nGOT THE TOP MODEL!",
          "\n-------------------")
    return top_model_name

#folder = os.path.join(os.getcwd(), 'MultiColorMaize_PipelineOutput')
#fileslist = os.listdir(folder)


# ls_files_wanted = []
def get_topfeatures(top_model_name_input):
    """
    Retrieves the feature importance DataFrame for the specified top model.

    Args:
        top_model_name_input (str): Name of the top-performing model.

    Returns:
        pandas.DataFrame: Feature importance DataFrame for the specified top model.
    """
    print('\n*****************************************************************')
    print("\n------------------------------------------",
          "\nGETTING THE TOP FEATURES OF THE TOP MODEL!",
          "\n-------------------------------------------")

    top_model_name = top_model_name_input.replace(" ", "_")
    len_model = len(top_model_name)
    # print("\nTop model name:", top_model_name)
    # print(len_model)
    print('\n--- Loading in the file associated to the top models features '
          'for ---')
    for filename in os.listdir():
        # print(filename)
        # print(filename[-5:])
        if (filename[:len_model] == str(top_model_name)) & \
                (filename[-3:] == "csv"):  # for scaling
            # ls_files_wanted.append(filename)
            df_feats_topmodel = pd.read_csv(filename)
            print(df_feats_topmodel.head())
            # print("FOUND:", filename)
        else:
            pass

    print("\n-----------------------",
          "\nTOP FEATURES LOADED IN!",
          "\n------------------------")

    return df_feats_topmodel


def feature_selected_inputfile(df_feats_topmodel, og_x_df, num_topfeats=20):
    """
    Selects the top features from the original input dataframe based on the
    feature importance dataframe.

    Args:
        df_feats_topmodel (pandas.DataFrame): Feature importance dataframe.

        og_x_df (pandas.DataFrame): Original input dataframe.

        num_topfeats (int, optional): Number of top features to select.
        Defaults to 20.

    Returns:
        pandas.DataFrame: New dataframe containing only the selected top
        features.
    """
    print('\n*****************************************************************')
    print("\n------------------------------------------",
          "\nGENERATING NEW INPUT FILE!",
          "\n-------------------------------------------")

    print("\nFeature importance dataframe shape:", df_feats_topmodel.shape)

    ls_features = df_feats_topmodel['Feature'].values.tolist()
    # print('\nList of features\n', ls_features)

    ls_top_feats = ls_features[0:num_topfeats]
    print("\nTop", num_topfeats, "Features:\n")
    for feature in ls_top_feats:
        print(feature)

    print("\nLength of top features list is:", len(ls_top_feats))

    new_x_featselected = og_x_df.filter(ls_top_feats, axis=1)
    print("\nOriginal Input dataframe features shape:",
          og_x_df.shape,
          "\nNew dataframe after feature selections shape:",
          new_x_featselected.shape)

    print("\n--------------------------",
          "\nNEW INPUT FILE GENERATED!",
          "\n--------------------------")
    return new_x_featselected
