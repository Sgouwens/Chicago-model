from Chicagodata import Chicago

# Filling a series of 7 days, ranging from 1st to 7th of august 2019 
# For interpolation, we use a whole week before and after the masked dates.
fill_days = 7
with_number = 2
first_date_interval = pd.to_datetime('8-1-2019')

chicago = Chicago()
chicago.fit_predict_xgb_model(first_date_interval, fill_days, with_number)

data_agg = chicago.data_train
data_agg = data_agg[(41.855 <= data_agg["Latitude"]) & (data_agg["Latitude"] <= 41.885)]
data_agg = data_agg[(-87.705 <= data_agg["Longitude"]) & (data_agg["Longitude"] <= -87.615)]
data_agg = data_agg[data_agg["Date"].dt.year >= 2019]

g = sns.FacetGrid(data_agg, col='Longitude', row='Latitude', sharey=False)
g.map(sns.scatterplot, "Date", "Count", color="b")
g.map(sns.scatterplot, "Date", "Count_prediction", color="r")

