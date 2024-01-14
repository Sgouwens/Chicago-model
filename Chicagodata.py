import pandas as pd
import polars as pl
import numpy as np
import os
import re

from itertools import product

# How does groupby and sort work?

class Chicagodata():
    """This class is designed to prepare the data on criminality in Chicago. It also
    preprocesses the data such that is the correct input for the model in R."""
    
    def __init__(self):
        """Loading and preparing the Chicago criminality data. This data has been downloaded from 
              https://www.kaggle.com/datasets/adelanseur/crimes-2001-to-present-chicago"""
              
        self.data = self.prepare_data()
        
    def load_dataset(self):
        """Loading the csv"""
        return pd.read_csv('./data/Crimes_-_2001_to_Present.csv')
    
    def prepare_data(self):
        """Performing the desired datamanipulation. This consists of selecting the right columns. 
        Neglecting the specific time in a day and only looking at blocks rather than exact 
        coordinates"""
        mesh = 0.03
        
        primary_type = ["THEFT", "NARCOTICS", "ASSAULT", "ROBBERY"]
        
        data = self.load_dataset()
        data = data[data["Primary Type"].isin(primary_type)]
        data = data[~data["Domestic"]]
        
        data["Date"] = pd.to_datetime(data["Date"], format='%m/%d/%Y %I:%M:%S %p').dt.floor("d")
        
        data = data[(2005 <= data["Date"].dt.year) & (data["Date"].dt.year <= 2019)]
        data = data[(41.705 <= data["Latitude"]) & (data["Latitude"] <= 41.975)]
        data = data[(-87.825 <= data["Longitude"]) & (data["Longitude"] <= -87.585)]
        
        data["Longitude_disc"] = pd.cut(data["Longitude"], bins=np.arange(-87.9, -87.5, mesh))
        data["Latitude_disc"] = pd.cut(data["Latitude"], bins=np.arange(41.6, 42.15, mesh))
        
        data["Latitude"] = data["Latitude_disc"].apply(lambda x: x.mid)
        data["Longitude"] = data["Longitude_disc"].apply(lambda x: x.mid)
        
        data = data.value_counts(['Date', 'Primary Type', 'Latitude', 'Longitude'])
        data = data.reset_index(name="Count")
                
        data = data.dropna()
        
        data = self.complete_data(data)
        
        data = data.sort_values(by=['Latitude', 'Longitude','Primary Type', 'Date'])
                
        if not os.path.exists("./data/chicago-crime-preprocessed.csv"):
            data.to_csv('./data/chicago-crime-preprocessed.csv', index=False)
            
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data = data.drop("Date", axis=1)
        
        return data
    
    @staticmethod
    def complete_data(df):
        """When no criminal records are made, data is missing. In reality there is no record
        because no criminal activity is found. Such missing rows will get count of 0."""
        all_combinations = list(product(df['Date'].unique(), 
                                        df['Primary Type'].unique(), 
                                        df['Latitude'].unique(), 
                                        df['Longitude'].unique()))
                                                                        
        df_complete = pd.DataFrame(all_combinations, 
                                   columns=['Date', 'Primary Type', 'Latitude', 'Longitude'])

        df = pd.merge(df_complete, df, 
                      on=['Date', 'Primary Type', 'Latitude', 'Longitude'],
                      how='left')
        
        df = df.fillna(0)
        
        return df

df = Chicagodata().data