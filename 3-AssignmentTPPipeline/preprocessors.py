import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var + '_NA'] = np.where(X[var].isnull(), 1, 0)
        return X


# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accommodate sklearn pipeline functionality
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.imputer_dict_ = {}
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        for var in self.variables:
            self.imputer_dict_[var] = X[var].median()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(self.imputer_dict_[var])

        return X


# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fiit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].str[0]
        return X


# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        self.encoder_dict_ = {}
        self.tol = tol

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        for var in self.variables:
            tmp = X.groupby(var)[var].count() / len(X)
            self.encoder_dict_[var] = tmp[tmp > self.tol].index
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.encoder_dict_[var]), X[var], 'Rare')

        return X


# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.dummies = {}

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # HINT: persist the dummy variables found in train set
        for var in self.variables:
            self.dummies[var] = pd.get_dummies(X[var], prefix=var, drop_first=True)

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for var in self.variables:
            # get dummies
            X = pd.concat([X, pd.get_dummies(X[var], prefix=var, drop_first=True)], axis=1)
            # drop original variables
            X.drop(labels=var, axis=1, inplace=True)

        # add missing dummies if any
        for var in self.variables:
            for d in self.dummies[var]:
                if d not in X.columns:
                    X[d] = 0

        return X
