""" multicolormaize created by Ally Schumacher """
import os
from pathlib import Path
from datapreprocessing import *
from runmodels import *
from featureselection import *


def create_output_dir(new_dir_name):
    """
    Creates new directory to output figures and files.

    Args:
        new_dir_name: (str) name of new directory

    Returns:
        str: path of the newly created directory

    Example:
        create_output_dir("results")
        Creates a directory named "results" in the current working directory
        Returns the path of the newly created directory: "./results"
    """
    output_directory = "./" + new_dir_name
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    return output_directory

# Provide the list of file paths to merge and preprocess
data_paths = [
    "../tutorial/widiv_2021drone_SilkandAntherColor.csv",
    # Add other file paths as needed
]

# Specify the merge column, y column, and columns to drop
merge_column = "GRIN"
y_column = "AntherColor"
columns_to_drop = ["SilkColor"]

X, y, cleaned_file_path = merge_and_preprocess_data(
    file_paths=data_paths,
    merge_column=merge_column,
    y_column=y_column,
    columns_to_drop=columns_to_drop
)

X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)
results_df, feature_importances = run_models(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             cleaned_file_path)
print("\nResults df:\n", results_df.head())
print("\nFeat importance df:\n", feature_importances.head())
plot_topfeatures()
top_model_name = get_top_model_name(results_df)
print("\nTop performing model was:\n", top_model_name)

# get the top features assocaited top model file
df_topmodel_featscore = get_topfeatures(top_model_name)
X_topfeatures = feature_selected_inputfile(df_topmodel_featscore, X)

# run top model again with feature selected dataframe

# get new traininig and testing set
X_train_feat, X_test_feat, y_train_feat, y_test_feat = split_dataset(
    X_topfeatures, y, test_size=0.2)
results_df_featselected, feature_importances_featselected = \
    featureselect_run_topmodel(X_train_feat,
                               X_test_feat,
                               y_train_feat,
                               y_test_feat,
                               cleaned_file_path,
                               top_model_name)
print("\nResults df:\n", results_df_featselected.head())
print("\nFeat importance df:\n", feature_importances_featselected.head())
featureselect_plot_topfeatures(top_model_name)

# Return to the original working directory
os.chdir("..")

print("\n********************************************************************")
print("\n-----------------------------------------------------------",
      "\nCONGRATULATIONS! YOU HAVE RAN THE PIPELINE SUCESSFULLY! :)",
      "\n-----------------------------------------------------------")
