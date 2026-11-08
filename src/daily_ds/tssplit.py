#%%
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

pd.options.plotting.backend = 'plotly'

#%%
rng = np.random.default_rng(0)
n = 2000
t = pd.date_range("2026-01-01", periods=n, freq="15min")
drift = np.cumsum(rng.normal(0, 0.05, n))        # slow sensor drift
temp = 60 + 5*np.sin(np.arange(n)/96*2*np.pi) + drift + rng.normal(0, 0.5, n)
df = pd.DataFrame({"temp": temp}, index=t)

df.plot(kind='line')

#%% Features use only the past: shift BEFORE rolling
df["lag1"] = df["temp"].shift(1)
df["lag4"] = df["temp"].shift(4)
df["roll_mean_8"] = df["temp"].shift(1).rolling(8).mean()
df["target"] = df["temp"].shift(-4)               # predict 1 h ahead
df = df.dropna()
X, y = df[["lag1", "lag4", "roll_mean_8"]], df["target"]

X.head()

#%% Scaler inside the pipeline -> it is fitted on each training fold only
model = make_pipeline(StandardScaler(),
                      RandomForestRegressor(n_estimators=100, random_state=0))

cvs = {"shuffled KFold": KFold(5, shuffle=True, random_state=0),
       "TimeSeriesSplit": TimeSeriesSplit(n_splits=5, gap=4)}  # gap = horizon
#%%

for name, cv in cvs.items():
    s = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(f"{name:16s} MAE = {s.mean():.3f} ± {s.std():.3f}")

for i, (tr, te) in enumerate(cvs["TimeSeriesSplit"].split(X)):
    print(i, X.index[tr[-1]], "->", X.index[te[0]], "n_train =", len(tr))
#%% Walk-forward forecast: train on each fold, predict every scan in the next block
preds = []
for tr, te in cvs["TimeSeriesSplit"].split(X):
    m = clone(model).fit(X.iloc[tr], y.iloc[tr])   # fresh copy per fold
    preds.append(pd.Series(m.predict(X.iloc[te]), index=X.index[te]))
pred = pd.concat(preds)
print("Walk-forward MAE:", round((pred - y.loc[pred.index]).abs().mean(), 3))

# A prediction made at scan time t is for t + 1 h -> plot it at the time it forecasts
pred_at_target = pred.copy()
pred_at_target.index = pred.index + pd.Timedelta("1h")

pd.DataFrame({"actual temp": df["temp"],
              "predicted (made 1 h earlier)": pred_at_target}).plot(kind="line")

#%% Final model: the same setup, trained once on every labelled row
final_model = clone(model).fit(X, y)

# The last 4 scans were dropped from X because their 1 h-ahead target isn't known yet.
# Those are the ones to forecast, so rebuild their features from the raw series.
raw = pd.Series(temp, index=t)
X_new = pd.DataFrame({"lag1": raw.shift(1),
                      "lag4": raw.shift(4),
                      "roll_mean_8": raw.shift(1).rolling(8).mean()}).iloc[-4:]

forecast = pd.Series(final_model.predict(X_new),
                     index=X_new.index + pd.Timedelta("1h"), name="forecast temp")
forecast   # forecasts for the next hour, indexed by the time they predict

# To reuse later without retraining:
# import joblib; joblib.dump(final_model, "temp_model.joblib")
# final_model = joblib.load("temp_model.joblib")
# %%
