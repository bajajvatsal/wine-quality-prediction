# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
#  # Red Wine Quality
# %% [markdown]
#  ## Import the libraries

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split,GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')
sns.set()

# %% [markdown]
# ## Import the dataset

# %%
df_wine = pd.read_csv(r"winequality-red.csv", sep=";")

# %% [markdown]
#  ## Exploratory Data Analysis

# %%
df_wine


# %%
df_wine.describe()

# %% [markdown]
#  Counting, unique values in the quality feature of the dataset

# %%
unique, counts = np.unique(df_wine["quality"], return_counts=True)
print(np.asarray((unique, counts)).T)


# %%
df_wine["quality"]


# %%
df_wine.isnull().sum()


# %%
corr = df_wine.corr()
fig, ax = plt.subplots(figsize=(11, 11))
mask = np.triu(df_wine.corr())
sns.heatmap(corr, ax=ax, cmap=sns.diverging_palette(230, 20, as_cmap=True), square=True, mask=mask, linewidths=.5)
plt.show()


# %%
sns.distplot(df_wine["quality"])
# sns.distplot

# %% [markdown]
#  According to the above graph, it is obvious that the target feature "quality" is highly biased so we had to do over-sampling or undersampling.<br>And since the data is limited we had to do oversampling.
# %% [markdown]
#  ## Data Preprocessing
# %% [markdown]
# Splitting the data in input variables and output variables

# %%
X, y = df_wine.iloc[:, :-1].values, df_wine.iloc[:, -1].values

# %% [markdown]
#  Encoding the wine quality

# %%
encoder = LabelEncoder()
df_wine["quality"] = encoder.fit_transform(df_wine["quality"])


# %%
unique, counts = np.unique(y, return_counts=True)
print(np.asarray((unique, counts)).T)

# %% [markdown]
#  ### Oversampling
#   Since we know that the data is highly biased on some values in the quality feature, we are using oversampling method to solve this problem

# %%
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
counter = Counter(y)


# %%
unique, counts = np.unique(y, return_counts=True)
print(np.asarray((unique, counts)).T)


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)


# %%
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %% [markdown]
# ### Scaling the training features
#  Since there are no categorical features, we just had to scale the training features and they all have data-type float64

# %%
scaling_list = df_wine.select_dtypes(["int64", "float64"]).columns
scaling_list


# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
#  ## Model Training
#  Using many different ML models from linear regressor to xgboost regressor
# 
# %% [markdown]
# ### Linear Regressor

# %%
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred).round()
y_pred = y_pred.astype(int)
y_test = np.array(y_test)
# targets = ["Class 2","Class 3","Class 4"]
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### Support Vector Regressor with GridSearchCV

# %%
regressor = SVR()
parameters = [{"kernel": ["linear"], "gamma": [i for i in np.arange(0.1, 1.0, 0.1)]},
              {"kernel": ["rbf"], "gamma":[i for i in np.arange(0.1, 1.0, 0.1)]}]
#               {"kernel": ["poly"], "degree":[i for i in range(1,4)],"gamma":[i for i in np.arange(0.1,0.5,0.1)]}]
grid_search = GridSearchCV(estimator=regressor,
                           param_grid=parameters,
                           scoring="accuracy",
                           n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Params: ", grid_search.best_params_)

# %% [markdown]
#  ### SVR model training

# %%
regressor = SVR(gamma=0.1, kernel="linear")
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred).round()
y_pred = y_pred.astype(int)
y_test = np.array(y_test)
print(classification_report(y_test, y_pred))
set(y_test) - set(y_pred)

# %% [markdown]
#  ### Random Forest Regressor

# %%
regressor = RandomForestRegressor(max_depth=50, n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred).round()
print(classification_report(y_test, y_pred, labels=np.unique(y_pred)))

# %% [markdown]
# ### Decision Tree Regressor with GridSearchCV

# %%
regressor = DecisionTreeRegressor()
parameters = [{"criterion": ["mse"], "max_features":["auto", "sqrt", "log2"]},
              {"criterion": ["friedman_mse"], "max_features":["auto", "sqrt", "log2"]},
              {"criterion": ["mae"], "max_features":["auto", "sqrt", "log2"]}]
grid_search = GridSearchCV(estimator=regressor,
                           param_grid=parameters,
                           scoring="accuracy",
                           n_jobs=-1)
grid_search.fit(X_train, y_train)
best_acc = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy {:.2f} %".format(best_acc*100))
print("Best Params: ", grid_search.best_params_)


# %%
regressor = DecisionTreeRegressor(criterion="friedman_mse", max_features="auto")
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred).round()
print(classification_report(y_test, y_pred, labels=np.unique(y_pred)))

# %% [markdown]
#  ### K Nearest Neighbors

# %%
regressor = KNeighborsRegressor(n_neighbors=7)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred).round()
print(classification_report(y_test, y_pred, labels=np.unique(y_pred)))

# %% [markdown]
#  ### XgBoost Regression

# %%
model = XGBRegressor()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores


# %%
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))


