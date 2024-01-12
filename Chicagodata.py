import pandas as pd
import polars as pl
import numpy as np
import os

class Chicagodata():
    """This class is designed to prepare the data on criminality in Chicago. It also
    preprocesses the data such that is the correct input for the model in R."""
    
    def __init__(self):
        
        print(f"Loading the Chicago criminality data. This data has been downloaded from \
              https://www.kaggle.com/datasets/adelanseur/crimes-2001-to-present-chicago")
              
        self.data = self.prepare_data()
        
        if not os.path.exists("./data/chicago-crime-preprocessed.csv"):
            Chicagodata().export_data_for_R()

    def load_dataset(self):
        return pd.read_csv('./data/Crimes_-_2001_to_Present.csv')
    
    def prepare_data(self):
        
        primary_type = ["THEFT", "NARCOTICS", "ASSAULT", 
                        "ROBBERY", "CRIMINAL TRESPASS", 
                        "WEAPONS VIOLATION", "PROSTITUTION"]
        
        data = self.load_dataset()
        data = data[data["Primary Type"].isin(primary_type)]
        data = data[~data["Domestic"]]
        data = data[["Date", "Primary Type", "Latitude", "Longitude"]]
        data = data.dropna()
        
        data["Date"] = pd.to_datetime(data["Date"], format='%m/%d/%Y %I:%M:%S %p').dt.floor("d")
        
        return data
                
    def export_data_for_R(self):
        """Exports data used in the spatial Poisson model in R"""
            
        mesh = 0.03
        data = self.data

        data["Longitude_disc"] = pd.cut(data["Longitude"], bins=np.arange(-87.9, -87.5, mesh))
        data["Latitude_disc"] = pd.cut(data["Latitude"], bins=np.arange(41.6, 42.15, mesh))

        data_agg = data.value_counts(['Date', 'Primary Type', 'Longitude_disc', 'Latitude_disc'])\
            .reset_index(name="Count")

        data_agg["Latitude_mid"] = data_agg["Latitude_disc"]\
            .apply(lambda x: float(re.findall(r'[-+]?\d*\.\d+|\d+', str(x))[0]) + mesh/2)
        data_agg["Longitude_mid"] = data_agg["Longitude_disc"]\
            .apply(lambda x: float(re.findall(r'[-+]?\d*\.\d+|\d+', str(x))[0]) + mesh/2)

        data_agg.to_csv('./data/chicago-crime-preprocessed.csv', index=False)
    