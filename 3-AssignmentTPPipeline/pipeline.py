from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config

titanic_pipe = Pipeline(
    [
        ('categorical_imputer', pp.CategoricalImputer(config.CATEGORICAL_VARS)),
        ('missing_indicator', pp.MissingIndicator(config.NUMERICAL_VARS)),
        ('numerical_imputer', pp.NumericalImputer(config.NUMERICAL_VARS)),
        ('extract_first_letter', pp.ExtractFirstLetter(config.CABIN)),
        ('rare_label_categorical_encoder', pp.RareLabelCategoricalEncoder(0.05, config.CATEGORICAL_VARS)),
        ('categorical_encoder', pp.CategoricalEncoder(config.CATEGORICAL_VARS)),
        ('scaler', StandardScaler()),
        ('linear_model', LogisticRegression(C=0.0005, random_state=0))
    ]
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
)
