import unittest
from Regression import DataAndFitLinearRegression
import pandas as pd


class TestRegression(unittest.TestCase):
    def setUp(self):
        self.regression = DataAndFitLinearRegression() 

    def test_analyze_and_fit(self):
        result = self.regression.analyze_and_fit()
        self.assertIn('summary_dict', result)
        self.assertIn('regression_dict', result)

        summary_dict = result['summary_dict']
        self.assertIn('statistics', summary_dict)
        self.assertIn('data_frame', summary_dict)
        self.assertIn('num_of_observations', summary_dict)

        regression_dict = result['regression_dict']['model_params']
        self.assertIn('Intercept', regression_dict)
        self.assertIn('Bedroom', regression_dict)
        self.assertIn('Space', regression_dict)
        self.assertIn('Room', regression_dict)
        self.assertIn('Lot', regression_dict)
        self.assertIn('Tax', regression_dict)
        self.assertIn('Bathroom', regression_dict)
        self.assertIn('Garage', regression_dict)
        self.assertIn('Condition', regression_dict)

    def test_prediction(self):
        result = self.regression.analyze_and_fit()
        pred = result['regression_dict']['price_prediction']
        self.assertIsInstance(pred, float)
    
    def test_invalid_data(self):
        # Test with invalid data path
        self.regression.train_data = './invalid_path.csv'
        with self.assertRaises(FileNotFoundError):
            self.regression.analyze_and_fit()

        with self.assertRaises(KeyError):
            self.regression._build_model(pd.DataFrame())  # Empty DataFrame

    def test_valid_data_path(self):
        # Test with a valid data path
        self.regression.train_data = './data/realest.csv'
        try:
            result = self.regression.analyze_and_fit()
            self.assertIn('summary_dict', result)
            self.assertIn('regression_dict', result)
        except Exception as e:
            self.fail(f"analyze_and_fit() raised {type(e).__name__} unexpectedly: {e}")

        



if __name__ == '__main__':
    unittest.main(exit=False)