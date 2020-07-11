# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:27:14 2020

@author: wyckliffe
"""
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from fbprophet import Prophet

train_df = pd.read_csv('datasets/train.csv')

# data descriptions
# almost a million observation
# 1115 unique stores

# Id: transaction ID (combination of Store and date)
# Store: unique store Id
# Sales: sales/day, this is the target variable
# Customers: number of customers on a given day
# Open: Boolean to say whether a store is open or closed (0 = closed, 1 = open)
# Promo: describes if store is running a promo on that day or not
# StateHoliday: indicate which state holiday (a = public holiday, b = Easter holiday, c = Christmas, 0 = None)
# SchoolHoliday: indicates if the (Store, Date) was affected by the closure of public schools

info_df = pd.read_csv('datasets/store.csv')

# StoreType: categorical variable to indicate type of store (a, b, c, d)
# Assortment: describes an assortment level: a = basic, b = extra, c = extended
# CompetitionDistance (meters): distance to closest competitor store
# CompetitionOpenSince [Month/Year]: provides an estimate of the date when competition was open
# Promo2: Promo2 is a continuing and consecutive promotion for some stores (0 = store is not participating, 1 = store is participating)
# Promo2Since [Year/Week]: date when the store started participating in Promo2
# PromoInterval: describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

# check for missing data
sns.heatmap(train_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")

# plot some bar plots
train_df.hist(bins = 30, figsize = (20,20), color = 'r')

#how many stores are open and closed!
closed_train_df        = train_df[train_df['Open'] == 0]
open_train_df          = train_df[train_df['Open'] == 1]

# Count the number of stores that are open and closed
print("Total =", len(train_df))
print("Number of closed stores =", len(closed_train_df))
print("Number of open stores =", len(open_train_df))


# only keep open stores and remove closed stores
train_df = train_df[train_df['Open'] == 1]

#drop the open column since it has no meaning now
train_df.drop(['Open'], axis=1, inplace=True)

# check for any missing data in the store information dataframe!
sns.heatmap(info_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")

# It seems like if 'promo2' is zero, 'promo2SinceWeek', 'Promo2SinceYear', and 'PromoInterval' information is set to zero
# There are 354 rows where 'CompetitionOpenSinceYear' and 'CompetitionOpenSinceMonth' is missing
# set these values to zeros
str_cols = ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']

for str in str_cols:
    info_df [str].fillna(0, inplace = True)

sns.heatmap(info_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")

# There are 3 rows with 'competitionDistance' values missing fill them up with with average values of the 'CompetitionDistance' column
info_df['CompetitionDistance'].fillna(info_df['CompetitionDistance'].mean(), inplace = True)

sns.heatmap(info_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")

info_df.hist(bins = 30, figsize = (20,20), color = 'r')


#merge both data frames together based on 'store'
train_all_df = pd.merge(train_df, info_df, how = 'inner', on = 'Store')

train_all_df.to_csv('datasets/test.csv', index=False)


correlations = train_all_df.corr()['Sales'].sort_values()
print(correlations)


correlations = train_all_df.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)


# separate the year and put it into a separate column
train_all_df['Year'] = pd.DatetimeIndex(train_all_df['Date']).year
train_all_df['Month'] = pd.DatetimeIndex(train_all_df['Date']).month
train_all_df['Day'] = pd.DatetimeIndex(train_all_df['Date']).day


axis = train_all_df.groupby('Month')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'r')
axis.set_title('Average Sales Per Month')

plt.figure()
axis = train_all_df.groupby('Month')[['Customers']].mean().plot(figsize = (10,5), marker = '^', color = 'b')
axis.set_title('Average Customers Per Month')


ax = train_all_df.groupby('Day')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'r')
axis.set_title('Average Sales Per Day')

plt.figure()
ax = train_all_df.groupby('Day')[['Customers']].mean().plot(figsize = (10,5), marker = '^', color = 'b')
axis.set_title('Average Sales Per Day')

axis = train_all_df.groupby('DayOfWeek')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'r')
axis.set_title('Average Sales Per Day of the Week')

plt.figure()
axis = train_all_df.groupby('DayOfWeek')[['Customers']].mean().plot(figsize = (10,5), marker = '^', color = 'b')
axis.set_title('Average Customers Per Day of the Week')


fig, ax = plt.subplots(figsize=(20,10))
train_all_df.groupby(['Date','StoreType']).mean()['Sales'].unstack().plot(ax=ax)


plt.figure(figsize=[15,10])

plt.subplot(211)
sns.barplot(x = 'Promo', y = 'Sales', data = train_all_df)

plt.subplot(212)
sns.barplot(x = 'Promo', y = 'Customers', data = train_all_df)

plt.subplot(211)
sns.violinplot(x = 'Promo', y = 'Sales', data = train_all_df)

plt.subplot(212)
sns.violinplot(x = 'Promo', y = 'Customers', data = train_all_df)



def sales_prediction(Store_ID, sales_df, periods):
  # Function that takes in the data frame, storeID, and number of future period forecast
  # The function then generates date/sales columns in Prophet format
  # The function then makes time series predictions

  sales_df = sales_df[ sales_df['Store'] == Store_ID ]
  sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date': 'ds', 'Sales':'y'})
  sales_df = sales_df.sort_values('ds')

  model    = Prophet()
  model.fit(sales_df)
  future   = model.make_future_dataframe(periods=periods)
  forecast = model.predict(future)
  figure   = model.plot(forecast, xlabel='Date', ylabel='Sales')
  figure2  = model.plot_components(forecast)


sales_prediction(10, train_all_df, 60)



def sales_prediction(Store_ID, sales_df, holidays, periods):
  # Function that takes in the storeID and returns two date/sales columns in Prophet format
  # Format data to fit prophet

  sales_df = sales_df[ sales_df['Store'] == Store_ID ]
  sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date': 'ds', 'Sales':'y'})
  sales_df = sales_df.sort_values('ds')

  model    = Prophet(holidays = holidays)
  model.fit(sales_df)
  future   = model.make_future_dataframe(periods = periods)
  forecast = model.predict(future)
  figure   = model.plot(forecast, xlabel='Date', ylabel='Sales')
  figure2  = model.plot_components(forecast)

school_holidays = train_all_df[train_all_df['SchoolHoliday'] == 1].loc[:, 'Date'].values


# Get all the dates pertaining to state holidays
state_holidays = train_all_df [ (train_all_df['StateHoliday'] == 'a') | (train_all_df['StateHoliday'] == 'b') | (train_all_df['StateHoliday'] == 'c')  ].loc[:, 'Date'].values

state_holidays = pd.DataFrame({'ds': pd.to_datetime(state_holidays),
                               'holiday': 'state_holiday'})

school_holidays = pd.DataFrame({'ds': pd.to_datetime(school_holidays),
                                'holiday': 'school_holiday'})

# concatenate both school and state holidays
school_state_holidays = pd.concat((state_holidays, school_holidays))

# Let's make predictions using holidays for a specific store
sales_prediction(6, train_all_df, school_state_holidays, 60)