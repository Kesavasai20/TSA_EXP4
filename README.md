# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 30/9/2025



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


path = 'Clean_Dataset.csv'

data = pd.read_csv(path)
if 'price' not in data.columns:
    raise ValueError("Dataset must contain a 'price' column. Columns found: " + ", ".join(data.columns))


date_col = None
for c in ['date', 'Date', 'DATE', 'timestamp', 'Timestamp']:
    if c in data.columns:
        date_col = c
        break

if date_col:
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col]).reset_index(drop=True)
    data = data.sort_values(date_col).reset_index(drop=True)
    data = data.set_index(date_col)
    print(f"Using datetime index from column '{date_col}'.")
else:
    print("No date column detected; using integer index.")


data['price'] = pd.to_numeric(data['price'], errors='coerce')
data = data.dropna(subset=['price']).copy()
print(f"Records after cleaning: {len(data)}")


if date_col:

    data_daily = data['price'].resample('D').mean().dropna()
    X = data_daily
    print(f"Aggregated to daily series length: {len(X)} (resample='D').")
else:
    X = data['price'].reset_index(drop=True)

if len(X) > 10000:
    fit_len = 5000

    X_fit = X.iloc[-fit_len:].copy().reset_index(drop=True)
    print(f"Using the most recent {fit_len} observations for model fitting.")
else:
    X_fit = X.copy()

plt.rcParams['figure.figsize'] = [12, 4]
if hasattr(X, 'index') and not isinstance(X.index, pd.RangeIndex):
    plt.plot(X.index[:2000], X.values[:2000])
else:
    plt.plot(X.values[:2000])
plt.title('Original Flight Price Data (first 2000 points)')
plt.xlabel('Date' if date_col else 'Index')
plt.ylabel('Price')
plt.show()


if len(X) > 5000:
    step = max(1, len(X)//5000)
    X_for_adf = X.iloc[::step].reset_index(drop=True)
    print(f"Series large: using subsample of length {len(X_for_adf)} for ADF.")
else:
    X_for_adf = X.copy()

adf_res = adfuller(X_for_adf.dropna(), autolag='AIC', maxlag=20)
adf_stat, adf_pvalue = adf_res[0], adf_res[1]
print(f"ADF Statistic (on subsample): {adf_stat:.4f}, p-value: {adf_pvalue:.4f}")

d = 0
if adf_pvalue > 0.05:
    print("Series appears non-stationary. Differencing once (d=1).")
    X_diff = X_fit.diff().dropna()
    d = 0
else:
    print("Series appears stationary (d=0).")
    X_diff = X_fit.copy()


max_lags = min(40, int(len(X_fit)/4))
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plot_acf(X_diff, lags=max_lags, ax=plt.gca())
plt.title('ACF (used series for modelling)')
plt.subplot(2,1,2)
plot_pacf(X_diff, lags=max_lags, ax=plt.gca(), method='ywm')
plt.title('PACF (used series for modelling)')
plt.tight_layout()
plt.show()

results = {}
for order in [(1,d,1),(2,d,2)]:
    try:
        print(f"Fitting ARIMA{order} on subset ...")
        model = ARIMA(X_fit, order=order)
        fitted = model.fit(method_kwargs={"maxiter":200})
        print(f"Fitted ARIMA{order}. AIC: {fitted.aic:.2f}")
        print(fitted.summary().tables[1])
        results[order] = fitted
    except Exception as e:
        print(f"Failed to fit ARIMA{order}: {e}")
        results[order] = None


def simulate_from_fitted(fitted, order_name, N=1000):
    if fitted is None:
        print(f"No fitted model for {order_name}; skipping simulation.")
        return
    params = fitted.params
    ar_keys = [k for k in params.index if k.startswith('ar.L')]
    ma_keys = [k for k in params.index if k.startswith('ma.L')]
    ar_coeffs = [params[k] for k in sorted(ar_keys, key=lambda x: int(x.split('L')[1]))] if ar_keys else []
    ma_coeffs = [params[k] for k in sorted(ma_keys, key=lambda x: int(x.split('L')[1]))] if ma_keys else []
    print(f"{order_name} AR coeffs: {ar_coeffs}")
    print(f"{order_name} MA coeffs: {ma_coeffs}")

    ar = np.r_[1, -np.array(ar_coeffs)] if len(ar_coeffs) > 0 else np.array([1.0])
    ma = np.r_[1, np.array(ma_coeffs)] if len(ma_coeffs) > 0 else np.array([1.0])
    arma_proc = ArmaProcess(ar, ma)
    try:
        sim = arma_proc.generate_sample(nsample=N)
        plt.figure(figsize=(10,3))
        plt.plot(sim)
        plt.title(f"Simulated {order_name} series (first {min(500,N)} points shown)")
        plt.xlim([0, min(500,N)])
        plt.show()

        plt.figure(figsize=(8,3))
        plot_acf(sim, lags=40, ax=plt.gca())
        plt.title(f"ACF of simulated {order_name}")
        plt.show()

        plt.figure(figsize=(8,3))
        plot_pacf(sim, lags=40, ax=plt.gca(), method='ywm')
        plt.title(f"PACF of simulated {order_name}")
        plt.show()
    except Exception as e:
        print(f"Simulation failed for {order_name}: {e}")

simulate_from_fitted(results.get((1,d,1)), f"ARIMA(1,{d},1)")
simulate_from_fitted(results.get((2,d,2)), f"ARIMA(2,{d},2)")

print("Completed. Notes: model fits were done on a recent subset for speed. Change resample frequency or fitting subset as needed.")

```
## OUTPUT:
<img width="1797" height="641" alt="image" src="https://github.com/user-attachments/assets/9be75d13-7ade-4332-a1f5-7662331a9767" />
<img width="1265" height="413" alt="image" src="https://github.com/user-attachments/assets/4e4c96e6-11e6-4706-a61e-20a498d3e602" />
<img width="1374" height="624" alt="image" src="https://github.com/user-attachments/assets/44aeb5e8-646f-4238-a36c-fb831be3f4ea" />
<img width="1208" height="590" alt="image" src="https://github.com/user-attachments/assets/ff0c9f7a-5bee-468f-b169-0048213a2849" />
<img width="884" height="222" alt="image" src="https://github.com/user-attachments/assets/6d42ecb7-df8c-414b-a206-534af9bc20d3" />


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
