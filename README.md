# Wine Quality Prediction Using Different Regression Models

This is classic problem of predicting red wine quality with the given physicochemical characteristics like - pH level, density, sulphur levels etc.
You can find more about the data on the [data-description](winequality.names) file.

The dataset and the notebook also uploaded at [Kaggle](https://www.kaggle.com/vatsalbajaj/wine-quality-prediction)

Check more on the notebook [Github Notebook](wine-quality.ipynb) / [Web-nbviewer](to be added)
or
[Python Script](wine-quality.py)

## Problem

Predicting the red-wine quality on the scale 1 to 10.

## Parameter for the classification

The parameters given in the dataset to predict sale price are -

1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol

## Target-Variable

Quality (score between 0 and 10), based on sensory data.

## Data Collection

Dataset published on [UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)

## Exploratory Data Analysis

EDA in this project used to find the relation between the input variables, dtype of the input variables for scaling/encoding as applicable and detect the sampling anomaly in the target variable.

### Visualizations

1. Density distribution of the wine quality
2. Correlation heat-map

## Data Preprocessing

Data Preprocessing involves processing data through algorithms/processes in order to make data train the model
Steps involved are -

1. Oversampling of the target feature, as the volume of data is highly concentrated only on two values
2. Label encoding the target feature
3. Scaling of the training data

## Model

Here we are using different types of regression models -

1. Linear Regressor
2. Support Vector Regressor
3. Decision Tree Regressor
4. K Nearest Neighbors Regressor
5. Random Forest Regressor
6. XgBoost Regressor

In the above models we are also using hyperparameter tuning using GridSearchCV.
