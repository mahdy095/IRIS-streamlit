#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data manipulation and analysis
import pandas as pd
import numpy as np
import scipy as sp
import os
import tabulate
import scipy.stats as stats
# pd.set_option('display.max_columns',100)

# Data visualization
from matplotlib import pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly as pl

# Machine learning
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder, PolynomialFeatures, RobustScaler, \
    MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import datasets, neighbors, metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import svm
from sklearn.svm import SVR
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.naive_bayes import GaussianNB
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import preprocessing
import random
import re
import datetime
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import make_scorer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import pairwise_distances
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.preprocessing import RobustScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import KernelPCA
import statsmodels.api as sm
from sklearn.feature_selection import f_regression, SelectKBest, mutual_info_regression
import math
from sklearn.neural_network import MLPRegressor, MLPClassifier
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# In[ ]:


st.write(
    """
    ## This is a simple Iris flower prediction app
    This app predicts **Iris flower** type
    """
)

df = sns.load_dataset('iris')

st.sidebar.header('User input parameters')

sepal_length = st.sidebar.slider('Sepal_length', df['sepal_length'].min(), df['sepal_length'].max(),
                                 df['sepal_length'].mode().values[0]
                                 )

sepal_width = st.sidebar.slider('Sepal_width', df['sepal_width'].min(), df['sepal_width'].max(),
                                df['sepal_width'].mode().values[0])

petal_length = st.sidebar.slider('Petal_length', df['petal_length'].min(), df['petal_length'].max(),
                                 df['petal_length'].mode().values[0])

petal_width = st.sidebar.slider('Petal_width', df['petal_width'].min(), df['petal_width'].max(),
                                df['petal_width'].mode().values[0])

user_input = {
    'sepal_length': sepal_length,
    'sepal_width': sepal_width,
    'petal_length': petal_length,
    'petal_width': petal_width,
}

st.write(""" 
## User input parameters
""")
user_input_df = pd.DataFrame(data=user_input.items(), columns=['Parameters', 'Values'])
st.table(user_input_df)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X = df.drop(columns=['species'])
df['species'] = df['species'].apply(
    lambda x: 0 if x == df['species'].unique()[0] else 1 if x == df['species'].unique()[1] else 2)

y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True, stratify=y, random_state=42)

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', MLPClassifier())
])

grid = GridSearchCV(
    estimator=pipe,
    param_grid={
        'model__hidden_layer_sizes': [(50, 50), (100, 100)],
        'model__activation': ['relu'],
        'model__alpha': [0.0001],
        'model__learning_rate': ['constant'],
        'model__learning_rate_init': [0.001],
        'model__max_iter': [1000],
        'model__solver': ['adam'],
    },
    cv=3,
    refit=True,
    n_jobs=-1,
    scoring='f1'
)

grid.fit(X_train, y_train)
pred = grid.predict(X_test)
prop = grid.predict_proba(X_test)

# Create an input feature vector for prediction based on user input
input_features = np.array(list(user_input.values())).reshape(1, -1)  # Reshape to match the model's input shape

# Load your trained machine learning model (grid.best_estimator_ contains the best model from GridSearchCV)

# Make predictions and probabilities based on user input
predictions = grid.predict(input_features)
class_probabilities = grid.predict_proba(input_features)

classes = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
# Display predictions and class probabilities
st.write("## Predicted Iris Flower Species")
st.write("Predicted Species:", classes[predictions[0]])

st.write("## Class Probabilities")
st.write("Setosa Probability:", class_probabilities[0][0])
st.write("Versicolor Probability:", class_probabilities[0][1])
st.write("Virginica Probability:", class_probabilities[0][2])

# Create a DataFrame for the bar chart
prop_dict = {
    'Setosa Probability': [class_probabilities[0][0]],
    'Versicolor Probability': [class_probabilities[0][1]],
    'Virginica Probability': [class_probabilities[0][2]]
}

prop_df = pd.DataFrame(prop_dict)

# Create and display the bar chart
st.subheader("Class Probability Chart")
fig = px.bar(x=prop_df.columns.tolist(), y=prop_df.values[0].tolist(),
             log_y=True, text_auto=True, color=prop_df.values[0].tolist(), title='Probability Distribution')
st.plotly_chart(fig)


