""" ne fonctionne que sur notebook : %matplotlib inline # sets the backend of matplotlib to the 'inline' backend"""
# import numpy as np
# import pandas as pd 
import matplotlib.pyplot as plt # matplotlib's plotting framework
import scipy.stats as stats
# probability distributions as well as a growing library of statistical functions
import sklearn.linear_model as linear_model
"""methods intended for regression in which
the target value is expected to be a linear combination of the input variables"""
import seaborn as sns # Visualize A pandas Dataframe
import xgboost as xgb
"""open-source software library which
provides the gradient boosting framework"""
from sklearn.model_selection import KFold
"""Each fold is then used once as a validation while
the k - 1 remaining folds form the training set"""
from IPython.display import HTML, display
# to consider something as HTML
from sklearn.manifold import TSNE
# visualize high-dimensional data
from sklearn.cluster import KMeans
# clusters the data into K number of clusters
from sklearn.decomposition import PCA
"""Finding the directions of maximum variance in high-dimensional data and
project it onto a smaller dimensional subspace while retaining most of the information"""
from sklearn.preprocessing import StandardScaler
"""Standardize features by removing the mean and scaling to unit variance"""

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/train.csv')

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']