import unittest
import os
from multicolormaize import create_output_dir

class TestCreateOutputDir(unittest.TestCase):

    def test_create_output_dir(self):
        new_dir_name = "test_dir"
        output_directory = create_output_dir(new_dir_name)
        self.assertTrue(os.path.isdir(output_directory))

if __name__ == '__main__':
    unittest.main()
