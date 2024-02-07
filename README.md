# Estimating thefts using XGBoost

In this notebook we will apply the XGBoost algoritm to model the expected criminality levels when a certain event did not occur. In Chicago there is an annual festival during which many thefts happen. The question is how many (arrests for) theft can be attributed to this festival.

Theft numbers are treated as time series. The dates of the festival are removed from the data, using surrounding points, we model the number of thefts that are expected to happen when the event never took place. If a model is trained on data in which the festival dates are not present, it will not learn anything about this festival. The model is then applied to the festival dates.

In (Wang, 2022)* the authors describe a method of timeseries forecasting using XGBoost. They do this by turning forecasting into a supervised learning problem. Given a timeseries $X_t$, one can regress $X_t$ as a function of its previous values:

$$X_t = f(X_{t-1},\dots, X_{t-h})$$

where $f$ is a function made of boosted trees. The idea behind this notebook is to extend this idea to predicting values between two datapoints:

$$X_t = f(X_{t-h_1},\dots, X_{t+h_2})$$

Or in this special case where the festival lasts multiple days, we want to impute multiple values of the time series. Let $T={\tau_a,\dots,\tau_b}$ be the indices time series points of multiple days: we compute for $\tau_0\in T$ the number of thefts according to

$$X_{\tau_0} = f(X_{\tau_a-h_1},\dots, X_{\tau_a-1}, \dots X_{\tau_b+1},\dots, X_{\tau_b+h_2})$$

Without too many specific details we continue with the analysis.

* https://ieeexplore.ieee.org/abstract/document/9058617

