import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CATEGORICAL_COLS = ['season']
MODEL = LinearRegression
CLF_MODEL = LogisticRegression


def prepare_data(data):
    df_categorical = data[CATEGORICAL_COLS]
    dummies = pd.get_dummies(df_categorical)
    data[dummies.columns] = dummies
    return data.drop(columns=df_categorical.columns)


def propensity(data, x_cols, T, model=CLF_MODEL):
    # Preprocess
    # Use standard scaling
    # Apply Logistic Regression on X, predict t
    # Return model's probabilities
    data = data.copy()
    data = prepare_data(data)

    X = data[x_cols]
    t = data[T]

    steps = [('scaler', StandardScaler()),
             ('clf', model(random_state=0))]
    pipe_lr = Pipeline(steps)
    clf = pipe_lr.fit(X, t)

    return clf.predict_proba(X)[:, 1]


def plot_propensity_hist(t, prop):
    kwargs = dict(histtype='stepfilled', alpha=0.7, bins=40)

    fig, ax = plt.subplots(1)

    ax.hist(prop[t], label='T = 1 (lost last game)', **kwargs)
    ax.hist(prop[~t], label='T = 0 (won last game)', **kwargs)
    plt.legend()

    ax.set_xlabel('Propensity')
    ax.set_ylabel('Count')
    ax.set_title("Propensity Histogram According to Treatment")

## IPW


def ate(data, prop, T, Y):
    # Calculate IPW formula (tutorial 8)
    left = (((data[T] * data[Y]) / prop).sum())
    right = (((1 - data[T]) * data[Y])/ (1 - prop)).sum()
    return (1/len(prop)) * (left - right)


def ipw(data, x_cols, T, Y, model=CLF_MODEL):
    prop = propensity(data, x_cols, T, model)
    ate_hat_ipw = ate(data, prop, T, Y)
    return ate_hat_ipw


## S-Learner


class InteractionsNoT(BaseEstimator, TransformerMixin):
    def __init__(self, T):
        self.poly = PolynomialFeatures(interaction_only=True, include_bias=False)
        self.T = T

    def fit(self, X, y):
        dummies_columns = [col for col in X.columns if 'season_' in col]

        self.poly.fit(X.drop(columns=[self.T] + dummies_columns), y)  # Perform interactions on X without T
        return self

    def transform(self, X):
        X = X.copy()
        dummies_columns = [col for col in X.columns if 'season_' in col]
        dummies_values = X[dummies_columns]
        t = X[self.T]
        X = self.poly.transform(X.drop(columns=[self.T] + dummies_columns))
        X = np.column_stack((X, dummies_values, t))
        return X


def s_learner_model(data, Y, model=MODEL):
    # Preprocess
    # Use standard scaling
    # Apply Linear Regression on X with t, predict y
    # Return model
    data = data.copy()
    data = prepare_data(data)
    X = data.drop(columns=[Y])
    y = data[Y]

    steps = [('scaler', StandardScaler()),
             ('clf', model())]
    pipe_lr = Pipeline(steps)
    clf = pipe_lr.fit(X, y)

    return clf


def s_learner_model_with_interactions(data, T, Y, model=MODEL):
    # Preprocess
    # Use standard scaling
    # Apply Linear Regression on X (+interactions) with t, predict y
    # Return model
    data = data.copy()
    data = prepare_data(data)
    X = data.drop(columns=[Y])
    y = data[Y]

    steps = [('interactions', InteractionsNoT(T)),
             ('scaler', StandardScaler()),
             ('clf', model())]
    pipe_lr = Pipeline(steps)
    clf = pipe_lr.fit(X, y)

    return clf


def calc_ate_s_learner(data, T, Y, clf):
    # Use S-learner (tutorial 8)
    data = data.copy()
    data = prepare_data(data).drop(columns=[Y])
    f1 = data
    f0 = data.copy()
    f0[T] = 0

    return (clf.predict(f1) - clf.predict(f0)).mean()


def s_learner(data, T, Y, use_interactions=False, model=MODEL):
    # Call S-learner with or without interactions
    clf = s_learner_model(data, Y, model=model) if not use_interactions else s_learner_model_with_interactions(data, T, Y, model=model)
    return calc_ate_s_learner(data, T, Y, clf)


## T-learner


def t_learner_model(data, t, T, Y, model=MODEL):
    # Preprocess and take only data with treatment t
    # Use standard scaling
    # Apply Linear Regression on X with t, predict y
    # Return model
    data = data.copy()
    data = prepare_data(data)
    data = data[data[T] == t]
    X = data.drop(columns=[Y, T])
    y = data[Y]

    steps = [('scaler', StandardScaler()),
             ('clf', model())]
    pipe_lr = Pipeline(steps)
    clf = pipe_lr.fit(X, y)

    return clf


def calc_ate_t_learner(data, T, Y, clf0, clf1):
    # Use T-learner (tutorial 8)
    data = data.copy()
    data = prepare_data(data)
    data_pred = data.drop(columns=[Y, T])

    return (clf1.predict(data_pred) - clf0.predict(data_pred)).mean()


def t_learner(data, T, Y, model=MODEL):
    # Call S-learner with or without interactions
    clf0 = t_learner_model(data, 0, T, Y, model=model)
    clf1 = t_learner_model(data, 1, T, Y, model=model)
    return calc_ate_t_learner(data, T, Y, clf0, clf1)

## Matching


def matching_1NN(data, T, Y):
    # Preprocess
    # Prepare data pairs
    # Calculate ITEs (lesson 3)
    # Calculate Average
    data = prepare_data(data)

    data_T0 = data[data[T] == 0]
    data_T0 = data_T0.drop(columns=[T])
    data_T1 = data[data[T] == 1]
    data_T1 = data_T1.drop(columns=[T])

    ITEs = []
    dists = cdist(np.array(data_T1.drop(columns=[Y])).astype(float), np.array(data_T0.drop(columns=[Y])).astype(float))
    for i, j in enumerate(dists.argmin(axis=1)):
        ITE_i = data_T1[Y].iloc[i] - data_T0[Y].iloc[j]
        ITEs.append(ITE_i)

    return np.mean(ITEs)


def matching_general(data, T, Y, k=10):
    # Preprocess
    # Prepare data near groups
    # Calculate ITEs (lesson 3)
    # Calculate Average
    data = prepare_data(data)

    data_T0 = data[data[T] == 0]
    data_T0 = data_T0.drop(columns=[T])
    data_T1 = data[data[T] == 1]
    data_T1 = data_T1.drop(columns=[T])

    ITEs = []
    i_dict = {}  # J(i)
    dists = cdist(np.array(data_T1.drop(columns=[Y])).astype(float), np.array(data_T0.drop(columns=[Y])).astype(float))
    k_nearest = np.argpartition(dists, k)[:, :k]
    for i, close_inds in enumerate(k_nearest):
        ITE_i = data_T1[Y].iloc[i] - data_T0[Y].iloc[close_inds].mean()
        ITEs.append(ITE_i)

    return np.mean(ITEs)

