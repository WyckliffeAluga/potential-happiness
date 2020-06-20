# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:07:24 2020

@author: wyckliffe
"""

import plotly.express as px
import numpy as np
import pandas as pd


url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

data = pd.read_csv(url, delimiter=',', header='infer')

data = data.loc[
    data['Country/Region'].isin(['United Kingdom', 'US', 'Italy', 'Brazil', 'India'])
    & data['Province/State'].isna()]
data.rename(
    index=lambda x: data.at[x, 'Country/Region'], inplace=True)
df1 = data.transpose()
df1 = df1.drop(['Province/State', 'Country/Region', 'Lat', 'Long'])
df1 = df1.loc[(df1 != 0).any(1)]
df1.index = pd.to_datetime(df1.index)
df1 = df1.diff() #day on day changes

