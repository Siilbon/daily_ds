#%%
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_process import ArmaProcess
import plotly.graph_objects as go
import plotly.express as px

pd.options.plotting.backend = 'plotly'
#%%

rng = np.random.default_rng(0)
n = 2000
# Loop temperature: AR(2) disturbance + slow ramp, 1-min samples
dist = ArmaProcess(ar=[1, -1.2, 0.5]).generate_sample(n, distrvs=rng.standard_normal)
t = np.arange(n)
temp = pd.Series(80 + 0.002 * t + dist,
                 index=pd.date_range("2026-01-01", periods=n, freq="1min"))

coef = np.polyfit(t, temp, 1)          # [slope, intercept], t in minutes
trend = pd.Series(np.polyval(coef, t), index=temp.index)

fig = temp.plot(kind='line')
# The data is drawn as WebGL (scattergl), which always renders above plain scatter
# traces, so the fit must also be Scattergl to show on top.
fig.add_trace(go.Scattergl(x=trend.index, y=trend, mode="lines", name="linear fit",
                           line=dict(color="red", dash="dash")))
fig.add_annotation(x=trend.index[-1], y=trend.iloc[-1],
                   text=f"temp = {coef[1]:.2f} + {coef[0]:.4f}*t<br>"
                        f"({coef[0]*60:.3f} Deg/h)",
                   showarrow=True, arrowhead=1, ax=-80, ay=-50,
                   bgcolor="white", bordercolor="red")
fig
#%%

# Detrend before looking at correlation structure
resid = temp - trend

fig_hist = resid.plot(kind='hist')
fig_hist.add_vline(x=resid.mean())
fig_hist.add_annotation(x=0, y=40, ax=-110, text=f'{resid.mean():.3f}')
#%%

band = 1.96 / np.sqrt(len(resid))   # approx 95% white-noise band
out = pd.DataFrame({
    "acf_raw": acf(temp, nlags=10)[1:],
    "acf":     acf(resid, nlags=10)[1:],
    "pacf":    pacf(resid, nlags=10, method="ywm")[1:],
}, index=range(1, 11)).round(2)
out["pacf_sig"] = out["pacf"].abs() > band

out
# %%
# Long format (one row per lag x series) so plotly can make one panel per series
long = (out[["acf_raw", "acf", "pacf"]]
        .rename_axis("lag").reset_index()
        .melt(id_vars="lag", var_name="series", value_name="corr"))

fig_bar = px.bar(long, x="lag", y="corr", facet_col="series")
for y in (band, -band):   # white-noise band on every panel
    fig_bar.add_hline(y=y, line_dash="dot", line_color="red")
fig_bar.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))  # "series=acf" -> "acf"
fig_bar.update_xaxes(dtick=1)
fig_bar

#%% Alternative: all three series as grouped bars on one chart
fig_grouped = out[["acf_raw", "acf", "pacf"]].plot(kind="bar", barmode="group")
for y in (band, -band):
    fig_grouped.add_hline(y=y, line_dash="dot", line_color="red")
fig_grouped.update_xaxes(dtick=1, title="lag")
fig_grouped
# %%
