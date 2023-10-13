"""
SUPERVISED MODELS.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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

    #return training and testing sets
    return X_train, X_test, y_train, y_test


# define model algorithms that will be used
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Multilayer Perceptron': MLPClassifier(max_iter=1000),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis()
}

def run_models(X_train, X_test, y_train, y_test, results_filepath):
    """
    Trains and evaluates multiple machine learning models using the provided
    training and testing data. Outputs model performance metrics and feature
    importance values to files.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Testing data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Testing data labels.
        results_filepath (str): File path for saving the model performance
        metrics.

    Returns:
        tuple: A tuple containing the following:
            - results_df (DataFrame): DataFrame containing model performance
            metrics.
            - feature_importances (DataFrame): DataFrame containing feature
            importance values for each model.
    """
    print('\n*****************************************************************')
    print("\n-------------------------------------------------------------",
          "\nSTARING TO RUN MODELS AND GENERATE FEATURE IMPORANCE SCORES!",
          "\n-------------------------------------------------------------")

    # create output location for results after running models
    results = {}
    # generate model results
    print("\n--- Generating results ---\n")
    print("The following models will be ran on the dataset:\n")
    for model in models.keys():
        print(model)

    for name, model in models.items():
        # fit the model on training data
        model.fit(X_train, y_train)
        # generation predictions using the test data
        y_pred = model.predict(X_test)
        # output performance metrics
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-score': f1_score(y_test, y_pred, average='weighted'),
            'Confusion Matrix': confusion_matrix(y_test, y_pred)
        }

    print("\n--- Saving the file containing all model results ---\n")
    print("The perfomance metrics tracked are:\n",
          'Accuracy', 'Precision','Recall', 'F1-score','Confusion Matrix')

    # convert results dict to dataframe and save to an output file
    results_df = pd.DataFrame(results)
    results_df.to_csv("ModelResults_" + results_filepath)

    # generate feature importance values of each model to a separate output file
    print("\n--- Generating feature importance scores for each model ---")
    feature_imp = {}
    for name, model in models.items():
        # get feature importance for each model using built in attributes
        if hasattr(model, 'feature_importances_'):
            feature_imp[name] = {'Feature': X_train.columns,
                                 'Importance': model.feature_importances_}
            feature_importances_df = pd.DataFrame(
                {'Feature': X_train.columns,
                 'Importance': model.feature_importances_})
            feature_importances_df.sort_values(
                by='Importance', ascending=False, inplace=True)
            feature_importances_df.to_csv(
                f'{name.replace(" ", "_")}_feature_importances.csv',
                index=False)
        elif hasattr(model, 'coef_'):
            feature_imp[name] = {'Feature': X_train.columns,
                                 'Importance': abs(model.coef_).sum(axis=0)}
            feature_importances_df = pd.DataFrame(
                {'Feature': X_train.columns,
                 'Importance': abs(model.coef_).sum(axis=0)})
            feature_importances_df.sort_values(by='Importance',
                                               ascending=False, inplace=True)
            feature_importances_df.to_csv(
                f'{name.replace(" ", "_")}_feature_importances.csv',
                index=False)
    feature_importances = pd.DataFrame(feature_imp)

    print("\n-----------------------------------------------------------------",
          "\nCOMPLETED MODEL GENERATION WITH RESULTS AND FEATURE IMPORANTANCE!",
          "\n-----------------------------------------------------------------")
    return results_df, feature_importances


def plot_topfeatures():
    """
    Plots the top features and their importance scores for each model and
    saves to current directory

    Returns:
        Figures in current directory showing top features

    """
    # plot feature importance for each model
    print('\n*****************************************************************')
    print("\n-------------------------------------",
          "\nPLOTTING TOP FEATURES FOR EACH MODEL",
          "\n-------------------------------------")

    print("\n--- Generating and saving feature importance figures ---\n")

    for name in models.keys():
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
            print(f'{name.replace(" ", "_")} '
                  f'feature importances file not found.')

    print("\n----------------------------------------------",
          "\nEACH MODEL's TOP FEATURES HAVE BEEN COMPLETED!",
          "\n----------------------------------------------")


def featureselect_run_topmodel(X_train, X_test, y_train, y_test,
                               results_filepath, top_modelname):
    """
        Trains and evaluates the identified top model after feature selection.
        Outputs model performance metrics and feature importance values to files
        of new feature selected dataframe.

        Args:
            X_train (array-like): Training data features.
            X_test (array-like): Testing data features.
            y_train (array-like): Training data labels.
            y_test (array-like): Testing data labels.
            results_filepath (str): File path for saving the model performance
            metrics.
            top_modelname (str): Name of the top model to be evaluated.

        Returns:
            tuple: A tuple containing the following:
                - results_df (DataFrame): DataFrame containing model performance
                 metrics.
                - feature_importances (DataFrame): DataFrame containing feature
                importance values for the top model.
        """
    print('\n*****************************************************************')
    print("\n-------------------------------------------------------------",
          "\nRERUN MODEL USING ONLY TOP MODEL ON TOP FEATURE DATASET",
          "\n-------------------------------------------------------------")
    # train and evaluate models
    name = top_modelname.replace("_", " ")
    results = {}
    models[name].fit(X_train, y_train)
    y_pred = models[name].predict(X_test)
    print("\n--- Generating results using feature selected data ---")
    results[top_modelname] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-score': f1_score(y_test, y_pred, average='weighted'),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

    # output model performance metrics to file
    print("\n--- Saving new model results after feature selection ---")
    results_df = pd.DataFrame(results)
    results_df.to_csv("ModelResults_AfterFeatureSelection_" +
                      results_filepath)

    # output feature importance values of each model to a separate output file
    print("\n--- Re-Generating feature importance scores for top model ---")
    feature_imp = {}
    if hasattr(models[name], 'feature_importances_'):
        feature_imp[name] = {'Feature': X_train.columns,
                             'Importance': models[name].feature_importances_}
        feature_importances_df = pd.DataFrame(
            {'Feature': X_train.columns,
             'Importance': models[name].feature_importances_})
        feature_importances_df.sort_values(by='Importance',
                                           ascending=False, inplace=True)
        feature_importances_df.to_csv(
            f'{name.replace(" ", "_")}'
            f'_feature_importances_AfterFeatSelection.csv',
            index=False)
    elif hasattr(models[name], 'coef_'):
        feature_imp[name] = {'Feature': X_train.columns,
                             'Importance': abs(models[name].coef_).sum(axis=0)}
        feature_importances_df = pd.DataFrame(
            {'Feature': X_train.columns,
             'Importance': abs(models[name].coef_).sum(axis=0)})
        feature_importances_df.sort_values(by='Importance', ascending=False,
                                           inplace=True)
        feature_importances_df.to_csv(
            f'{name.replace(" ", "_")}'
            f'_feature_importances_afterFeatSelection.csv',
            index=False)
    feature_importances = pd.DataFrame(feature_imp)

    print("\n---------------------------------------------------------------",
          "\nNEW MODEL RESULTS AND TOP FEATURES MADE WITH NEW FEAT DATASET!",
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
    if models.keys() == top_modelname:
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
