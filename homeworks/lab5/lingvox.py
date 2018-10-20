import warnings
import warnings
warnings.filterwarnings('ignore', category=Warning)



import numpy as np
import pandas as pd

from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score, make_scorer
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.multioutput import ClassifierChain

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RandomizedLogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectKBest, chi2
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.preprocessing import scale, StandardScaler, RobustScaler

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import binarize, FunctionTransformer


def read_data_from_file(filename, shape):
    values = list()
    rows = list()
    cols = list()

    header = True
    for line in open(filename):
        if header:
            header = False
            continue
        row, col, value = [x for x in line.strip().split(',')]
        row, col = int(row), int(col)
        value = float(value)
        row -= 1
        col -= 1
        values.append(value)
        rows.append(row)
        cols.append(col)

    return sparse.csr_matrix((values, (rows, cols)), shape=shape)


def read_labels_from_file(filename, shape):
    labels = np.zeros(shape).astype(int)

    header = True
    for line in open(filename):
        if header:
            header = False
            continue
        row, indeces = line.strip().split(',')
        row = int(row) - 1
        indeces = [int(x) - 1 for x in indeces.split()]
        labels[row, indeces] = 1

    return labels


class MLScoreWrapper:
    
    def __init__(self, score_func, Y_ml):
        self.score_func = score_func
        self._preprocess(Y_ml)
        self.n_lables = Y_ml.shape[1]
        self.__name__ = score_func.__name__
    
    @property
    def n_classes(self):
        return len(self.ml2mc_dict)
    
    def _preprocess(self, Y_ml):
        multi_labels = []
        for i in range(len(Y_ml)):
            indeces = np.where(Y_ml[i, :] == 1)[0] + 1
            multi_labels.append(sorted(indeces))
        unique_labels, counts = np.unique(multi_labels, return_counts=True)
        
        self.ml2mc_dict = {}
        self.mc2ml_dict = {}

        for i, label in enumerate(unique_labels):
            self.ml2mc_dict[tuple(label)] = i
            self.mc2ml_dict[i] = tuple(label)
        
        self.Y_mc = []
        for label in multi_labels:
            self.Y_mc.append(self.ml2mc_dict[tuple(label)])
        
    def mc2ml(self, y):
        n_samples = len(y)
        
        Y = np.zeros((n_samples, self.n_lables), dtype=int)
        
        for i, cls in enumerate(y):
            Y[i, np.array(self.mc2ml_dict[cls]) - 1] = 1
        return Y
    
    def __call__(self, y_true, y_pred, **kwargs):
        y_true_ml = self.mc2ml(y_true)
        y_pred_ml = self.mc2ml(y_pred)
        
        return self.score_func(y_true_ml, y_pred_ml, **kwargs)


def get_train():
    return read_data_from_file('X_train.csv', (15000, 60000)).astype(float)


def get_y():
    return read_labels_from_file('Y_train.csv', (15000, 100))

