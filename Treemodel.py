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
        self.predicted = None
    
    def prepare_columns(self, backward, forward):
        """Given the data, in order to make future prediction into a supervised problem
        we need to change the dataframe such that we have columns: X_{t+1} as response and
        X_{t-h} ... X_{t} as predictor variables. When predicting in-between values, the 
        predictor variates are the ones in the past and in the future of predicted point."""
        
        # grouped_shift is not used. This results in overflow of data from one group to another.
        data_new = self.data
        columns = ["Count_lag_" + str(h) for h in range(-backward, forward+1) if not h==0]
        
        for name in columns:
            h = int(re.search('[-0-9]+', name).group()) 
            data_new[name] = data_new["Count"].shift(-h)
        
        self.data = data_new.dropna()
    
    @staticmethod
    def grouped_shift(self, group, lag):
        """Group-wise shifting preventing overflow from separate groups"""
        group['lagged_value'] = group['value'].shift(lag)
        return group
    
    def fit_xgb_model(self, h):
        
        X = self.data.drop("Count", axis=1)
        y = self.data[["Count"]]
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        
        model.fit(X, y)
        self.predicted = model.predict(X)
        
        return mean_squared_error(df2.data["Count"], self.predicted)
        
lag = 2
df2 = Treemodel(df)

df2.prepare_columns(lag, lag)
df2.fit_xgb_model(lag)

data_agg = df2.data
data_agg['pred'] = df2.predicted
data_agg['error'] = data_agg['Count'] - data_agg['pred']

data_agg["Date"] = pd.to_datetime(data_agg[['Year', 'Month', 'Day']])

data_agg = data_agg[(41.855 <= data_agg["Latitude"]) & (data_agg["Latitude"] <= 41.945)]
data_agg = data_agg[(-87.73 <= data_agg["Longitude"]) & (data_agg["Longitude"] <= -87.3)]

g = sns.FacetGrid(data_agg, col='Latitude', row='Longitude')
g.map(sns.scatterplot, "Date", "error")

# Now train model on data where actual data is missing

# df2 = df2[df2['Latitude'] == 41.885]
# df2 = df2[df2['Longitude'] == -87.615]
# df2 = df2[df2["Date"].isin(pd.to_datetime(['2005-01-01','2005-01-02', '2005-01-03', '2005-01-04']))]
# df2
