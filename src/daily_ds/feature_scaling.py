#%%
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import plotly.express as px

pd.options.plotting.backend = 'plotly'

#%%
rng = np.random.default_rng(0)
n = 2000
df = pd.DataFrame({
    "temp_K":   rng.normal(350, 15, n),   # big numbers, weak signal
    "vib_mm_s": rng.normal(2.0, 0.5, n),  # small numbers, strong signal
    "flow_m3h": rng.normal(120, 5, n),
})
df.head()
#%%
spikes = rng.random(n) < 0.02             # ~2% comms glitches
#%%

df.loc[spikes, "flow_m3h"] += rng.normal(0, 400, spikes.sum())
y = (df.vib_mm_s + 0.02*(df.temp_K-350)
     + rng.normal(0, 0.2, n) > 2.3).astype(int)

df.plot(kind='line')
#%%
px.histogram(y)

#%%

models = {
    "kNN raw":      KNeighborsClassifier(15),
    "kNN Standard": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "kNN Robust":   make_pipeline(RobustScaler(), KNeighborsClassifier(15)),
    "RF raw":       RandomForestClassifier(200, random_state=0),
}
for name, m in models.items():
    print(f"{name:13s} {cross_val_score(m, df, y, cv=5).mean():.3f}")
# %%

# %%
