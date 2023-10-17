""" multicolormaize created by Ally Schumacher """
import os
from pathlib import Path
from datapreprocessing import *
from runmodels import *
from featureselection import *


def create_output_dir(new_dir_name, results_filename):
    """
    Creates a new directory to output figures and files.

    Args:
        new_dir_name: (str) Name of the new directory.
        results_filename: (str) Name of the results file.

    Returns:
        str: Path of the newly created directory.

    Example:
        create_output_dir("results", "results.csv")
        Creates a directory named "results" in the current working directory.
        Returns the path of the newly created directory: "./results"
    """
    output_directory = "./" + new_dir_name
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Construct the results file path within the newly created directory
    results_filepath = os.path.join(output_directory, results_filename)

    return output_directory, results_filepath

# Provide the list of file paths to merge and preprocess
data_paths = [
    "../tutorial/widiv_2021drone_SilkandAntherColor.csv",
    # Add other file paths as needed
]

# Specify the merge column, y column, and columns to drop
merge_column = "GRIN"
y_column = "AntherColor"
columns_to_drop = ["SilkColor"]
alg_type = 'classification' # indicate 'classification' or 'regression'

# Define the results file name
results_filename = "results.csv"

# Create the output directory and get the results file path
output_directory, results_filepath = create_output_dir("results", results_filename)

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
                                             algorithm_type=alg_type,
                                             results_filepath=results_filepath)
print("\nFeature Importances:\n")
for model_name, feature_importance in feature_importances.items():
    print(f"Model: {model_name}")
    for feature, importance in zip(feature_importance['Feature'], feature_importance['Importance']):
        print(f"Feature: {feature}, Importance: {importance}")
    print()
plot_topfeatures()

top_model_name = get_top_model_name(results_df)

print("\nTop performing model was:\n", top_model_name)
# get the top features assocaited top model file
df_topmodel_featscore = get_topfeatures(top_model_name)
X_topfeatures = feature_selected_inputfile(df_topmodel_featscore, X)

# Run the top model again with the feature-selected dataframe
X_train_feat, X_test_feat, y_train_feat, y_test_feat = split_dataset(
    X_topfeatures, y, test_size=0.2)

# Call the featureselect_run_topmodel function with the global_models dictionary
results_df_featselected, feature_importances_featselected = featureselect_run_topmodel(
    X_train_feat,
    X_test_feat,
    y_train_feat,
    y_test_feat,
    cleaned_file_path,
    top_model_name
)


print("\nResults df:\n", results_df_featselected.head())
print("\nFeature Importances:\n")
for model_name, feature_importance in feature_importances_featselected.items():
    print(f"Model: {model_name}")
    for feature, importance in zip(feature_importance['Feature'], feature_importance['Importance']):
        print(f"Feature: {feature}, Importance: {importance}")
    print()
featureselect_plot_topfeatures(top_model_name)

# Return to the original working directory
os.chdir("..")

print("\n********************************************************************")
print("\n-----------------------------------------------------------",
      "\nCONGRATULATIONS! YOU HAVE RAN THE PIPELINE SUCESSFULLY! :)",
      "\n-----------------------------------------------------------")
