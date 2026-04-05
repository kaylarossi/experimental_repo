import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LinearRegression


class DataAndFitLinearRegression:
    def __init__(self):
        self.version = 1
        self.train_data = './data/realest.csv'
        self.inf_data = './inf_config.json'
        self.model = None
        self.feature_names = None

    def analyze_and_fit(self):
        data = pd.read_csv(self.train_data)
        summary_dict = self._build_summary_dict(data)
        data = self.__listwise_deletion(data)
        model, feature_names = self._build_model(data)
        regression_dict = self._build_regression_dict()
        return {
            'summary_dict': summary_dict,
            'regression_dict': regression_dict
        }


    def _build_summary_dict(self, data: pd.DataFrame):
        # return 3 elements - statistics, data_frame, num_of_observations
        # tax for all houses with 2 bathrooms and 4 bedrooms
        filtered = data[(data['Bathroom']==2) & (data['Bedroom']==4)]
        tax = filtered['Tax']
        statistics = [
            tax.mean(),
            tax.std(),
            tax.median(),
            tax.min(),
            tax.max()
        ]
        #data_frame - obs space>800 ordered by dec price
        data_frame = data[(data['Space']>800)]
        data_frame = data_frame.sort_values('Price', ascending=False)

        #number of observations for value of variable lot => 4th 5-quantile
        fourth_quantile = data['Lot'].quantile(0.8)
        num_of_observations = data[data['Lot'] >= fourth_quantile].shape[0]

        return {
            'statistics': statistics,
            'data_frame': data_frame,
            'num_of_observations': num_of_observations
        }
    
    def _build_model(self, data: pd.DataFrame):
        target = data['Price']
        X = data.drop(columns=['Price'])
        y = target
        feature_names = X.columns.tolist()
        model = LinearRegression()
        model.fit(X, y)
        self.model = model
        self.feature_names = feature_names
        return model, feature_names

    def _build_regression_dict(self):
        with open(self.inf_data, 'r') as f:
            inference_data = json.load(f)
        model_params = {
            "Intercept": round(float(self.model.intercept_), 2)
        }

        coeffs = self.model.coef_
        for name, coeff in zip(self.feature_names, coeffs):
            model_params[name] = round(float(coeff), 2)
        
        inference = pd.DataFrame([inference_data])[self.feature_names]

        price_prediction = round(float(self.model.predict(inference)[0]), 2)
        

        return {
            'model_params': model_params,
            'price_prediction': price_prediction
        }
    
    def __listwise_deletion(self, data: pd.DataFrame):
        # remove all rows with missing values
        return data.dropna()