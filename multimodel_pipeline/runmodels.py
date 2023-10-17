import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import Lasso

# Define the global models variable outside of any function
global models

def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        X (pandas.DataFrame or numpy.ndarray): The feature data.
        y (pandas.Series or numpy.ndarray): The target data.
        test_size (float or int): The proportion of the dataset to include
                                  in the test split.
        random_state (int or None): Random seed for reproducible results.

    Returns:
        tuple: A tuple (X_train, X_test, y_train, y_test) containing the
               train-test split.
    """
    global models  # Declare 'models' as a global variable
    print('\n*****************************************************************')
    print("\n--------------------------------------------",
          "\nSPLITTING DATA INTO TRAINING AND TEST SETS",
          "\n--------------------------------------------")

    # generate training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\n----------------------------",
          "\nDATA SUCCESSFULLY SPLIT!",
          "\n----------------------------")

    # return training and testing sets
    return X_train, X_test, y_train, y_test

def create_models(algorithm_type):
    """
    Create and return a dictionary of models based on the algorithm type.

    Args:
        algorithm_type (str): Type of algorithm ('classification' or 'regression').

    Returns:
        dict: A dictionary containing the model names as keys and the model instances as values.
    """
    global models  # Declare 'models' as a global variable

    if algorithm_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Support Vector Machine': SVC(),
            'Gaussian Naive Bayes': GaussianNB(),
            'Multilayer Perceptron': MLPClassifier(max_iter=1000),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            'Gradient Boosting Classifier': GradientBoostingClassifier(),
            'AdaBoost Classifier': AdaBoostClassifier(),
            'Extra Trees Classifier': ExtraTreesClassifier(),
            'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
            'XGBoost Classifier': XGBClassifier(),
            'LightGBM Classifier': LGBMClassifier()
        }
    elif algorithm_type == 'regression':
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree Regression': DecisionTreeRegressor(),
            'Random Forest Regression': RandomForestRegressor(),
            'Gradient Boosting Regressor': GradientBoostingRegressor(),
            'AdaBoost Regressor': AdaBoostRegressor(),
            'Extra Trees Regressor': ExtraTreesRegressor(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Elastic Net Regression': ElasticNet(),
            'Support Vector Machine Regressor': SVR(),
            'XGBoost Regressor': XGBRegressor(),
            'LightGBM Regressor': LGBMRegressor()
        }

    return models

def run_models(X_train, X_test, y_train, y_test, algorithm_type, results_filepath):
    """
    Trains and evaluates multiple machine learning models using the provided
    training and testing data. Outputs model performance metrics and feature
    importance values to files.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Testing data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Testing data labels.
        algorithm_type (str): Type of algorithm ('classification' or 'regression').
        results_filepath (str): File path for saving the model performance metrics.

    Returns:
        tuple: A tuple containing the following:
            - results_df (DataFrame): DataFrame containing model performance metrics.
            - feature_importances (DataFrame): DataFrame containing feature importance values for each model.
    """
    global models  # Declare 'models' as a global variable
    print('\n*****************************************************************')
    print("\n-------------------------------------------------------------",
          "\nSTARTING TO RUN MODELS AND GENERATE FEATURE IMPORTANCE SCORES!",
          "\n-------------------------------------------------------------")

    # Create output location for results after running models
    results = {}
    feature_importances = {}

    # Check if the y labels contain unique classes other than 0 and 1
    unique_classes = set(y_train.unique()) | set(y_test.unique())
    if set(unique_classes) != {0, 1}:
        # Map the unique classes to 0 and 1 and print the mapping
        class_mapping = {}
        for i, cls in enumerate(unique_classes):
            class_mapping[cls] = i
        y_train = y_train.map(class_mapping)
        y_test = y_test.map(class_mapping)
        print("Class mapping to 0 and 1:")
        for original_class, mapped_class in class_mapping.items():
            print(f"Class {original_class} is mapped to {mapped_class}.")

    models = create_models(algorithm_type)

    # Generate model results
    print("\n--- Generating results ---\n")
    print("The following models will be run on the dataset:\n")
    for name in models.keys():
        print(name)

    for name, model in models.items():
        # Fit the model on training data
        model.fit(X_train, y_train)
        # Generate predictions using the test data
        y_pred = model.predict(X_test)

        # Output performance metrics
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred) if algorithm_type == 'classification' else None,
            'Precision': precision_score(y_test, y_pred, average='weighted') if algorithm_type == 'classification' else None,
            'Recall': recall_score(y_test, y_pred, average='weighted') if algorithm_type == 'classification' else None,
            'F1-score': f1_score(y_test, y_pred, average='weighted') if algorithm_type == 'classification' else None,
            'Mean Squared Error': mean_squared_error(y_test, y_pred) if algorithm_type == 'regression' else None,
            'R-squared': r2_score(y_test, y_pred) if algorithm_type == 'regression' else None,
            'Confusion Matrix': confusion_matrix(y_test, y_pred) if algorithm_type == 'classification' else None
        }

        # Output feature importance values
        if hasattr(model, 'feature_importances_') and algorithm_type == 'classification':
            feature_importances[name] = {
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            }
        elif hasattr(model, 'coef_') and algorithm_type == 'classification':
            feature_importances[name] = {
                'Feature': X_train.columns,
                'Importance': abs(model.coef_).sum(axis=0)
            }

    # Convert results dict to DataFrame and save to an output file
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_filepath)  # Use the provided results_filepath

    # Generate feature importance values of each model to a separate output file
    if algorithm_type == 'classification':
        print("\n--- Generating feature importance scores for each model ---")
        for name, model in models.items():
            if name in feature_importances:
                feature_importances_df = pd.DataFrame(feature_importances[name])
                feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)
                feature_importances_df.to_csv(f'{name.replace(" ", "_")}_feature_importances.csv', index=False)

    print("\n-----------------------------------------------------------------",
          "\nCOMPLETED MODEL GENERATION WITH RESULTS AND FEATURE IMPORTANCE!",
          "\n-----------------------------------------------------------------")
    return results_df, feature_importances

def plot_topfeatures():
    """
    Plots the top features and their importance scores for each model and
    saves to the current directory

    Returns:
        Figures in the current directory showing top features
    """
    # plot feature importance for each model
    print('\n*****************************************************************')
    print("\n-------------------------------------",
          "\nPLOTTING TOP FEATURES FOR EACH MODEL",
          "\n-------------------------------------")

    print("\n--- Generating and saving feature importance figures ---\n")

    for name in models.keys():  # Use models instead of global_models
        try:
            model_featimp = pd.read_csv(
                f'{name.replace(" ", "_")}_feature_importances.csv')
            plt.figure()
            plt.title(f'{name} Feature Importance')
            plt.bar(x=model_featimp['Feature'][:20],
                    height=model_featimp['Importance'][:20],
                    label='Feature Importance Score')
            plt.xticks(rotation=90)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f'{name.replace(" ", "_")}_feature_importance_plot.png')
        except FileNotFoundError:
            print(f'{name.replace(" ", "_")} feature importances file not found.')

    print("\n----------------------------------------------",
          "\nEACH MODEL's TOP FEATURES HAVE BEEN COMPLETED!",
          "\n----------------------------------------------")


def featureselect_run_topmodel(X_train, X_test, y_train, y_test,
                               results_filepath, top_modelname):
    """
    Trains and evaluates the identified top model after feature selection.
    Outputs model performance metrics and feature importance values to files
    of the new feature-selected dataframe.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Testing data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Testing data labels.
        results_filepath (str): File path for saving the model performance metrics.
        top_modelname (str): Name of the top model to be evaluated.

    Returns:
        tuple: A tuple containing the following:
            - results_df (DataFrame): DataFrame containing model performance metrics.
            - feature_importances (DataFrame): DataFrame containing feature importance values for the top model.
    """
    print(
        '\n*****************************************************************')
    print(
        "\n-------------------------------------------------------------",
        "\nRERUN MODEL USING ONLY TOP MODEL ON TOP FEATURE DATASET",
        "\n-------------------------------------------------------------")

    results = {}

    # Retrieve the top model from the models dictionary
    top_model = models.get(top_modelname)

    if top_model is None:
        raise ValueError(
            f"Top model '{top_modelname}' not found in models dictionary")

    top_model.fit(X_train, y_train)
    y_pred = top_model.predict(X_test)

    print("\n--- Generating results using feature-selected data ---")
    results[top_modelname] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-score': f1_score(y_test, y_pred, average='weighted'),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

    # Output model performance metrics to file
    print("\n--- Saving new model results after feature selection ---")
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "ModelResults_AfterFeatureSelection_" + results_filepath)

    # Output feature importance values of the model to a separate output file
    print(
        "\n--- Re-Generating feature importance scores for top model ---")
    feature_imp = {}

    if hasattr(top_model, 'feature_importances_'):
        feature_imp[top_modelname] = {
            'Feature': X_train.columns,
            'Importance': top_model.feature_importances_
        }
        feature_importances_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': top_model.feature_importances_
        })
        feature_importances_df.sort_values(by='Importance',
                                           ascending=False, inplace=True)
        feature_importances_df.to_csv(
            f'{top_modelname.replace(" ", "_")}'
            f'_feature_importances_AfterFeatSelection.csv',
            index=False
        )
    elif hasattr(top_model, 'coef_'):
        feature_imp[top_modelname] = {
            'Feature': X_train.columns,
            'Importance': abs(top_model.coef_).sum(axis=0)
        }
        feature_importances_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': abs(top_model.coef_).sum(axis=0)
        })
        feature_importances_df.sort_values(by='Importance',
                                           ascending=False, inplace=True)
        feature_importances_df.to_csv(
            f'{top_modelname.replace(" ", "_")}'
            f'_feature_importances_afterFeatSelection.csv',
            index=False
        )

    feature_importances = pd.DataFrame(feature_imp)

    print(
        "\n---------------------------------------------------------------",
        "\nNEW MODEL RESULTS AND TOP FEATURES MADE WITH NEW FEATURE-DATASET!",
        "\n---------------------------------------------------------------")
    return results_df, feature_importances


def featureselect_plot_topfeatures(top_modelname):
    """
    Plots the top features and their importance scores for the specified top
    model after feature selection to figures in current directory.

    Args:
        top_modelname (str): Name of the top model.

    Returns:
        Nothing

    """
    print('\n*****************************************************************')
    print("\n------------------------------------------------------------",
          "\nPLOTTING TOP FEATURES FOR TOP MODEL AFTER FEATURE SELECTION",
          "\n------------------------------------------------------------")

    print("\n--- ReGenerating and saving feature importance figures ---")
    # plot feature importance for each model
    # for name in models.keys():
    if top_modelname in models:
        try:
            model_featimp = pd.read_csv(
                f'{top_modelname.replace(" ", "_")}'
                f'_feature_importances_AfterFeatSelection.csv')
            plt.figure()
            plt.title(
                f'{top_modelname} Feature Importance After Feature Selection')
            plt.bar(x=model_featimp['Feature'][:20],
                    height=model_featimp['Importance'][:20],
                    label='Feature Importance Score')
            plt.xticks(rotation=90)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f'{top_modelname.replace(" ", "_")}'
                f'_feature_importance_plot_AfterFeatSelection.png')
        except FileNotFoundError:
            print(
                f'{top_modelname.replace(" ", "_")} '
                f'feature importances after feature selection file not '
                f'found.')
    print("\n--------------------------------------------",
          "\nTOP MODEL USING TOP FEAUTURES FIGURES MADE!",
          "\n--------------------------------------------")
