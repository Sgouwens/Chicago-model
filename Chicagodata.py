import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import os
import re

from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# TODO: Add option to remove a specific part of the data before fitting the model. Adding this back after
# the fit and apply prediction. Make this a separate function that is then used in fit_predict_xgb_model.

# Per maand lijken day 2 en 3 te missen. 
# remove_data_interval werkt neit naar behoren

class Chicago():
    """This class is designed to prepare the data on criminality in Chicago. It also
    preprocesses the data such that is the correct input for the model in R."""
    
    def __init__(self, crime='THEFT'):
        """Loading and preparing the Chicago criminality data. This data has been downloaded from 
              https://www.kaggle.com/datasets/adelanseur/crimes-2001-to-present-chicago"""
              
        self.data = self.prepare_data(crime)
        self.crime = crime
        
        self.data_train = None
        self.data_predict= None
        self.model = None
        
    def load_dataset(self):
        """Loading the csv"""
        return pd.read_csv('./data/Crimes_-_2001_to_Present.csv')
    
    def prepare_data(self, crime):
        """Performing the desired datamanipulation. This consists of selecting the right columns. 
        Neglecting the specific time in a day and only looking at blocks rather than exact 
        coordinates"""
        mesh = 0.03
        
        primary_type = ["THEFT", "NARCOTICS", "ASSAULT", "ROBBERY"]
        
        data = self.load_dataset()
        data = data[data["Primary Type"]==crime].drop("Primary Type", axis=1)
        
        data = data[~data["Domestic"]]
        
        data["Date"] = pd.to_datetime(data["Date"], format='%m/%d/%Y %I:%M:%S %p').dt.floor("d")
        
        data = data[(2005 <= data["Date"].dt.year) & (data["Date"].dt.year <= 2019)]
        data = data[(41.705 <= data["Latitude"]) & (data["Latitude"] <= 41.975)]
        data = data[(-87.825 <= data["Longitude"]) & (data["Longitude"] <= -87.585)]
        
        data["Longitude_disc"] = pd.cut(data["Longitude"], bins=np.arange(-87.9, -87.5, mesh))
        data["Latitude_disc"] = pd.cut(data["Latitude"], bins=np.arange(41.6, 42.15, mesh))
        
        data["Latitude"] = data["Latitude_disc"].apply(lambda x: x.mid)
        data["Longitude"] = data["Longitude_disc"].apply(lambda x: x.mid)
        
        data = data.value_counts(['Date', 'Latitude', 'Longitude'])
        data = data.reset_index(name="Count")
                
        data = data.dropna()
        
        data = self.complete_data(data)
        
        data = data.sort_values(by=['Latitude', 'Longitude', 'Date'])
                
        if not os.path.exists("./data/chicago-crime-preprocessed.csv"):
            data.to_csv('./data/chicago-crime-preprocessed.csv', index=False)
            
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        #data = data.drop("Date", axis=1)
        
        return data
    
    @staticmethod
    def complete_data(data):
        """When no criminal records are made, data is missing. In reality there is no record
        because no criminal activity is found. Such missing rows will get count of 0."""
        all_combinations = list(product(data['Date'].unique(), 
                                        data['Latitude'].unique(), 
                                        data['Longitude'].unique()))
                                                                        
        df_complete = pd.DataFrame(all_combinations, 
                                   columns=['Date', 'Latitude', 'Longitude'])

        data = pd.merge(df_complete, data, 
                      on=['Date', 'Latitude', 'Longitude'],
                      how='left')
                
        return data.fillna(0)
    

    def remove_data_interval(self, first_date, k, m):
        """This function removes a date-interval to prepare data for interval predicting
        num_of_days (k) and with_num (m)"""
                
        date_interval = pd.date_range(start=first_date - pd.DateOffset(days=m), 
                                      end=first_date + pd.DateOffset(days=k+m))

        date_intervals = date_interval

        for i in range(-20, 2):
            date_intervals = date_intervals.union(date_interval + pd.DateOffset(years=i))
            
        date_interval = date_intervals
                
        data_temp = self.data_predict[~self.data_predict["Date"].isin(date_interval)]
                
        return data_temp
    
    def prepare_train_data(self, k, m):
        """Given the data, in order to make an inbetween prediction into a supervised problem
        we need to change the dataframe such that we have columns: X_{t+1} as response and
        X_{t-h} ... X_{t} as predictor variables. When predicting in-between values, the 
        predictor variates are the ones in the past and in the future of predicted point.
        
        In this application, a number (k) of consequetive days is filled using m days on both 
        sides to make a prediction. For point X_{t+k}, X_{t-m}..X_{t} and X_{t+k+1}..X{t+k+m}
        is used for prediction This function prepares this data."""
        
        # grouped_shift is not used. This results in overflow of data from one group to another.
        self.fill_with_days = (k, m) 
        
        data_train = self.data
                
        columns = ["Count_lag_" + str(h) for h in range(-m, k+m) if not h in range(k)]
                
        for name in columns:
            h = int(re.search('[-0-9]+', name).group()) 
            data_train.loc[:, name] = data_train.loc[:, "Count"].copy().shift(-h)
                
        return data_train
    
    @staticmethod
    def grouped_shift(self, group, lag):
        """Group-wise shifting preventing overflow from separate groups"""
        group['lagged_value'] = group['value'].shift(lag)
        return group
    
    def fit_predict_xgb_model(self, first_date, k, m): 
        """"""
        print("Computing data_predict")
        self.data_predict = self.prepare_train_data(k, m)
        print("Computing data_train")
        self.data_train = self.remove_data_interval(first_date, k, m)
        
        X_train = self.data_train.drop(["Count", "Date"], axis=1)
        y_train = self.data_train[["Count"]]
                
        
        print("Training model")
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.model.fit(X_train, y_train)
        
        print("Computing prediction")
        self.data_predict['Count_prediction'] = self.model.predict(
            self.data_predict.drop(["Count", "Date"], axis=1)
            )
                
        # print(f"The MSE of the model is {mean_squared_error(y, self.data['Count_prediction'])}")

if __name__=="__main__": 
    chicago = Chicago()
    chicago.fit_predict_xgb_model(first_date_interval, fill_days, with_number)