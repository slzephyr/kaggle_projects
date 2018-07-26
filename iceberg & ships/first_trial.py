import pandas as pd
import numpy as np

train = pd.read_json("~/Downloads/Kaggle_data/iceberg_classifier/train.json")
test = pd.read_json("~/Downloads/Kaggle_data/iceberg_classifier/test.json")

print("train data------------------\n")
print(train.head())
print("test data-------------------\n")
print(test.head())
