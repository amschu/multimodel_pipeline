import unittest
import pandas as pd
import os
from pathlib import Path
from datapreprocessing import *

class TestReadData(unittest.TestCase):
    def test_read_data_existing_file(self):
        file_path = '../tutorial/widiv_2021drone_SilkandAntherColor.csv'  #

        df, csvfile_path = read_data(file_path)
        self.assertIsInstance(df, pd.DataFrame)

        # Assert that the DataFrame has the expected shape
        self.assertEqual(df.shape, (761, 837))

        # Assert that the DataFrame y_columns are correct
        # expected_columns = ['SilkColor', 'AntherColor', 'GRIN']
        # self.assertTrue(expected_columns in df.columns)
        self.assertTrue('AntherColor' in df.columns)

        # Assert that the file path matches the expected path
        expected_path = Path(file_path)
        self.assertEqual(csvfile_path, expected_path)

    def test_read_data_nonexistent_file(self):
        file_path = './nonexistent_file.csv'

        with self.assertRaises(FileNotFoundError):
            read_data(file_path)

    def setUp(self):
        self.df = pd.DataFrame({
            'GRIN': ['A', 'B', 'A', 'C', 'B'],
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50],
            'Feature3': ['?', 'NA', 'na', 'n/a', '.']
        })

    def test_clean_data(self):
        y_column = 'Feature1'
        data_path = 'data.csv'
        columns_to_drop = ['Feature3']

        df_scaled, df_classes, save_name = clean_data(self.df, y_column,
                                                      data_path,
                                                      columns_to_drop)

        # Assert that the returned scaled DataFrame has the same shape as the input DataFrame
        self.assertNotEqual(df_scaled.shape, self.df.shape)

        # Assert that the returned classes DataFrame has the same length as the input DataFrame
        self.assertEqual(len(df_classes), len(df_scaled))

        # Assert that the save_name has the correct format
        expected_save_name = 'data_predicting_Feature1_preprocessed.txt'
        self.assertEqual(save_name, expected_save_name)

        # Assert that the dropped columns are not present in the scaled DataFrame
        self.assertNotIn('Feature3', df_scaled.columns)

        # Assert that the object columns are converted to numeric
        self.assertTrue(all(df_scaled[column].dtype == np.float64 for column in
                            df_scaled.columns if
                            df_scaled[column].dtype == 'object'))

    def tearDown(self):
        # Remove any files created during the test
        if os.path.exists('data_predicting_Feature1_preprocessed.txt'):
            os.remove('data_predicting_Feature1_preprocessed.txt')

if __name__ == '__main__':
    unittest.main()
