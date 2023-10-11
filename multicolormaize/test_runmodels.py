import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from runmodels import *


class TestRunModels(unittest.TestCase):
    def test_split_dataset(self):
        # Test case 1: Valid input with default arguments
        X = pd.DataFrame(np.random.randn(100, 5),
                         columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
        y = pd.Series(np.random.randint(0, 2, size=100))
        X_train, X_test, y_train, y_test = split_dataset(X, y)

        # Test if the shape of the training set is correct
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(y_train.shape[0], 80)

        # Test if the shape of the testing set is correct
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(y_test.shape[0], 20)

        # Test case 2: Valid input with custom test size
        test_size = 0.3
        X_train, X_test, y_train, y_test = split_dataset(X, y,
                                                         test_size=test_size)

        # Test if the shape of the training set is correct
        self.assertEqual(X_train.shape[0], 70)
        self.assertEqual(y_train.shape[0], 70)

        # Test if the shape of the testing set is correct
        self.assertEqual(X_test.shape[0], 30)
        self.assertEqual(y_test.shape[0], 30)

        # Test case 3: Invalid input with incompatible shapes
        X = pd.DataFrame(np.random.randn(100, 5),
                         columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
        y = pd.Series(np.random.randint(0, 2, size=50))

        # Test if the function raises a ValueError
        self.assertRaises(ValueError, split_dataset, X, y)

    def test_run_models(self):
        # Generate dummy data for testing
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        X_train = X.iloc[:100]
        X_test = X.iloc[100:]
        y_train = y[:100]
        y_test = y[100:]

        # Test if the function runs without errors
        results_df, feature_importances = run_models(X_train, X_test, y_train,
                                                     y_test, "results.csv")

        # Check if the results_df is a DataFrame
        self.assertTrue(isinstance(results_df, pd.DataFrame))

        # Check if the feature_importances is a DataFrame
        self.assertTrue(isinstance(feature_importances, pd.DataFrame))

        # Test if the output DataFrame has the correct shape
        self.assertEqual(results_df.shape[0], 5)  # Number of performance scores
        self.assertEqual(results_df.shape[1], 8)  # Number models

        # Test if the output DataFrame contains expected columns
        expected_columns = ['Logistic Regression', 'Decision Tree',
                            'Random Forest',
                            'K-Nearest Neighbors', 'Support Vector Machine',
                            'Gaussian Naive Bayes', 'Multilayer Perceptron',
                            'Linear Discriminant Analysis']
        self.assertListEqual(list(results_df.columns), expected_columns)

        # Test if the output DataFrame contains valid performance metrics
        self.assertTrue(all(results_df.iloc[0, :] >= 0))  # Accuracy >= 0
        self.assertTrue(all(results_df.iloc[1, :] >= 0))  # Precision >= 0
        self.assertTrue(all(results_df.iloc[2, :] >= 0))  # Recall >= 0
        self.assertTrue(all(results_df.iloc[3, :] >= 0))  # F1-score >= 0

        # Test if the output feature_importances DataFrame has the correct shape
        # Number of features
        self.assertEqual(X_train.shape[1],4)

        # Test if the feature_importances CSV files are created
        expected_files = ['Logistic_Regression_feature_importances.csv',
                          'Decision_Tree_feature_importances.csv',
                          'Random_Forest_feature_importances.csv',
                          'Linear_Discriminant_Analysis_feature_importances.csv']
        for name in expected_files:
            self.assertTrue(os.path.isfile(name))
            os.remove(name)

        # remove output file results from using this function
        os.remove("ModelResults_results.csv")



    def test_plot_topfeatures(self):
        # Create fake dictonary of models
        models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Support Vector Machine': SVC(),
            'Gaussian Naive Bayes': GaussianNB(),
            'Multilayer Perceptron': MLPClassifier(),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis()
        }

        # Create a fake CSV file for each model
        for name, model in models.items():
            model_csv = f'{name.replace(" ", "_")}_feature_importances.csv'
            model_data = {'Feature': ['A', 'B', 'C'],
                          'Importance': [0.2, 0.3, 0.5]}
            pd.DataFrame(model_data).to_csv(model_csv, index=False)


        # Check if the plot files were created for each model except the missing one
        for name in models.keys():
            plot_file = f'{name.replace(" ", "_")}_feature_importance_plot.png'
            if os.path.exists(plot_file):
                # Plot file should exist if the corresponding feature importances file exists
                self.assertTrue(os.path.exists(plot_file))

                # Remove the created plot file
                os.remove(plot_file)
            else:
                # Feature importances file is missing, plot file should not exist
                self.assertFalse(os.path.exists(plot_file))

            # Remove the created CSV file
            os.remove(f'{name.replace(" ", "_")}_feature_importances.csv')

if __name__ == '__main__':
    unittest.main()
