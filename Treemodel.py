import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import re

from Chicagodata import Chicagodata

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Treemodel():
    """"""
    
    def __init__(self, data, crime='THEFT'):
        self.data = data[data["Primary Type"]==crime].drop("Primary Type", axis=1)
        self.crime = crime
        self.model = None
        self.prediction = None
    
    def prepare_columns(self, k, m):
        """Given the data, in order to make an inbetween prediction into a supervised problem
        we need to change the dataframe such that we have columns: X_{t+1} as response and
        X_{t-h} ... X_{t} as predictor variables. When predicting in-between values, the 
        predictor variates are the ones in the past and in the future of predicted point.
        
        In this application, a number (k) of consequetive days is filled using m days on both 
        sides to make a prediction. For point X_{t+k}, X_{t-m}..X_{t} and X_{t+k+1}..X{t+k+m}
        is used for prediction This function prepares this data."""
        
        # grouped_shift is not used. This results in overflow of data from one group to another.
        data_new = self.data
        
        #columns = ["Count_lag_" + str(h) for h in range(-backward, forward+1) if not h==0]
        columns = ["Count_lag_" + str(h) for h in range(-m, k+m) if not h in range(k)]
        
        for name in columns:
            h = int(re.search('[-0-9]+', name).group()) 
            data_new[name] = data_new["Count"].shift(-h)
        
        self.data = data_new.dropna()
    
    @staticmethod
    def grouped_shift(self, group, lag):
        """Group-wise shifting preventing overflow from separate groups"""
        group['lagged_value'] = group['value'].shift(lag)
        return group
    
    def fit_predict_xgb_model(self, h):
        """"""
        X = self.data.drop("Count", axis=1)
        y = self.data[["Count"]]
        
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.model.fit(X, y)
        self.data['Count_prediction'] = self.model.predict(X)
        
        self.data = self.data.filter(regex='^(?!.*lag).*$', axis=1).copy()
        
        print(f"The MSE of the fitted model is {mean_squared_error(y, self.data['Count_prediction'])}")
    
      