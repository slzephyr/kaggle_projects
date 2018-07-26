# view the data

import pandas as pd
import numpy as np

df_train = pd.read_csv('/home/lei/Downloads/Kaggle_data/Africa_soil_property_prediction/training.csv')
df_sorted_test = pd.read_csv('/home/lei/Downloads/Kaggle_data/Africa_soil_property_prediction/sorted_test.csv')

# show the first 5 rows
print df_train.head()['SOC']
print df_sorted_test.head()

print df_train.info()
print df_sorted_test.info()


