import unittest
from pandas.testing import assert_frame_equal
from .featureselection import *


class TestFunctions(unittest.TestCase):
    def test_get_top_model_name(self):
        # Create a test DataFrame
        results_df = pd.DataFrame({
            '': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
            'Model A': [0.8, 0.75, 0.85, 0.8],
            'Model B': [0.82, 0.79, 0.81, 0.81],
            'Model C': [0.78, 0.76, 0.84, 0.79]
        })

        expected_output = 'Model B'
        top_model_name = get_top_model_name(results_df)
        self.assertEqual(top_model_name, expected_output)

    def test_get_topfeatures(self):
        # Create a test DataFrame
        df_feats_topmodel = pd.DataFrame({
            'Feature': ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
        })
        top_model_name_input = 'Model B'

        expected_output = pd.DataFrame({
            'Feature': ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
        })
        df_feats_topmodel = get_topfeatures(top_model_name_input)
        assert_frame_equal(df_feats_topmodel, expected_output)

    def test_feature_selected_inputfile(self):
        # Create test DataFrames
        df_feats_topmodel = pd.DataFrame({
            'Feature': ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
        })
        og_x_df = pd.DataFrame({
            'Feature 1': [1, 2, 3],
            'Feature 2': [4, 5, 6],
            'Feature 3': [7, 8, 9],
            'Feature 4': [10, 11, 12],
            'Other Feature': [13, 14, 15]
        })

        num_topfeats = 2

        expected_output = pd.DataFrame({
            'Feature 1': [1, 2, 3],
            'Feature 2': [4, 5, 6]
        })

        new_x_featselected = feature_selected_inputfile(df_feats_topmodel, og_x_df, num_topfeats)
        assert_frame_equal(new_x_featselected, expected_output)


if __name__ == '__main__':
    unittest.main()
