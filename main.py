import pandas as pd

# download the dataset from the given URL
data = pd.read_csv('/workspaces/The_Forage_Quant_Research/Nat_Gas (2).csv')
print(data.head())

# format the date column
data['Dates'] = pd.to_datetime(data['Dates'])
data.set_index('Dates', inplace=True)

print(data)

# check for missing values
print(data.isnull().sum())


# Train a linear regression model
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(data['Prices'], order=(5, 1, 0))
model = model.fit()
print(model.summary())

def estimate_price(data):
    """
    Estimate the next price given historical data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with 'Price' column and DatetimeIndex.

    Returns
    -------
    float
        Estimated price.
    """
    model = ARIMA(data['Prices'], order=(5, 1, 0))
    model = model.fit()
    return model.forecast()[0]

print(estimate_price(data))

# Find future prices
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=365, freq='D')

forecast  = model.predict(start = future_dates[0], end = future_dates[-1])
print(forecast)