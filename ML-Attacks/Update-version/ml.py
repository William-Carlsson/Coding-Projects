import os
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split


data = pd.read_csv('filtered_features.csv')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

train_data = TabularDataset('train.csv')
test_data = TabularDataset('test.csv')

predictor = TabularPredictor(label='label').fit(train_data=train_data)
predictions = predictor.predict(test_data)

# Print predictions
# print("Predictions on test data:")
# print(predictions)
# print("Feature importance:")
# print(predictor.feature_importance(test_data))

