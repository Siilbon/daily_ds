#%%
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

pd.options.plotting.backend = 'plotly'
#%%
class RollingFeatures(BaseEstimator, TransformerMixin):
    """Clip spikes (limits learned in fit), then add rolling mean/std."""
    def __init__(self, window=12, clip_q=0.99):
        self.window = window          # store as-is: no logic here
        self.clip_q = clip_q

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.upper_ = X.quantile(self.clip_q)   # learned state -> trailing _
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).clip(upper=self.upper_, axis=1)
        r = X.rolling(self.window, min_periods=1)
        return pd.concat([X, r.mean().add_suffix("_rmean"),
                          r.std().fillna(0).add_suffix("_rstd")], axis=1)

    def get_feature_names_out(self, input_features=None):
        b = list(self.feature_names_in_)
        return np.array(b + [f"{c}_rmean" for c in b] + [f"{c}_rstd" for c in b])

class LagFeatures(BaseEstimator, TransformerMixin):
    '''Lags the values at different times'''
    def __init__(self, lags=(1, 3, 6)) -> None:
        self.lags = lags

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        lagged = [X.shift(periods=lag).bfill().add_suffix(f'_lag{lag}') for lag in self.lags]
        return pd.concat([X, *lagged], axis=1)

    def get_feature_names_out(self, input_features=None):
        b = list(self.feature_names_in_)
        return np.array(b + [f'{c}_lag{lag}' for lag in self.lags for c in b])

#%%
# Synthetic 5-min process data
rng = np.random.default_rng(0)
idx = pd.date_range("2026-01-01", periods=500, freq="5min")
#%%
df = pd.DataFrame({
    "temp": 70 + 2*np.sin(np.arange(500)/30) + rng.normal(0, 0.3, 500),
    "pressure": 3.2 + rng.normal(0, 0.05, 500)}, index=idx)
y = 0.8 * df["temp"].rolling(12, min_periods=1).mean() + rng.normal(0, 0.2, 500)
df.head()
#%%
df.iloc[100, 0] = 500  # sensor spike in the inputs


df.plot(kind='line')
#%%

pipe = make_pipeline(RollingFeatures(window=12), Ridge())


pipe.fit(df.iloc[:400], y.iloc[:400])
print("R² test:", round(pipe.score(df.iloc[400:], y.iloc[400:]), 3))  # ~0.95
#%%
print(pipe[0].get_feature_names_out())
#%% Tunable like any built-in step:
gs = GridSearchCV(pipe, {"rollingfeatures__window": [6, 12, 24, 100]}, cv=TimeSeriesSplit())
gs.fit(df.iloc[:400], y.iloc[:400])

res = pd.DataFrame(gs.cv_results_)
cols = ["param_rollingfeatures__window", "mean_test_score", "std_test_score",
        "rank_test_score"] + [f"split{i}_test_score" for i in range(5)]
res[cols].sort_values("rank_test_score").round(3)
#%% Best setting, refit on all 400 training rows, scored on the hold-out
print(gs.best_params_)
print("Hold-out R²:", round(gs.score(df.iloc[400:], y.iloc[400:]), 3))
#%% Add lags after the rolling step and tune window + lags together
pipe_lag = make_pipeline(RollingFeatures(window=12), LagFeatures(lags=(1, 3, 6)), Ridge())

param_grid = {"rollingfeatures__window": [6, 12, 24],
              "lagfeatures__lags": [(1,), (1, 3, 6), (1, 6, 12)]}
gs_lag = GridSearchCV(pipe_lag, param_grid, cv=TimeSeriesSplit())
gs_lag.fit(df.iloc[:400], y.iloc[:400])

res_lag = pd.DataFrame(gs_lag.cv_results_)
# Rows = window, columns = lags (tuples cast to str so they work as labels)
res_lag.pivot_table(index="param_rollingfeatures__window",
                    columns=res_lag["param_lagfeatures__lags"].astype(str),
                    values="mean_test_score").round(3)
#%%
print(gs_lag.best_params_)
print("Hold-out R²:", round(gs_lag.score(df.iloc[400:], y.iloc[400:]), 3))

# %%
