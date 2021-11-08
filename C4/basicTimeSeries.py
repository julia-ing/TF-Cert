import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas.tests.io.formats.test_format import filepath_or_buffer
from tensorflow import keras

"""
fixed partitioning = train + validation + test
<metrics for evaluating performance>
1. errors = forecasts - actual
2. mse = np.square(errors).mean()  # mean squared error : 음수 값 제거
3. rmse = np.sqrt(mse)  # root : 기존 값과 같은 scale 유지를 위함
4. mae = np.abs(errors).mean()  # mean absolute error 
    -> keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy()
5. mape = np.abs(errors / x_valid).mean()  # mean absolute percentage error : 에러와 값의 비율
"""


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


# trend + seasonality + noise
def trend(time, slope=0):
    return slope * time


time = np.arange(4 * 365 + 1)
baseline = 10
series = trend(time, 0.1)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


baseline = 10
amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


noise_level = 5
noise = white_noise(time, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(time, noise)
plt.show()

series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


# autocorrelation 이란? scale 이 달라도 예측 가능한 모양을 따르는 데이터
def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ1 = 0.5
    φ2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += φ1 * ar[step - 50]
        ar[step] += φ2 * ar[step - 33]
    return ar[50:] * amplitude


def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += φ * ar[step - 1]
    return ar[1:] * amplitude


series = autocorrelation(time, 10, seed=42)
plot_series(time[:200], series[:200])
plt.show()

series = autocorrelation(time, 10, seed=42) + trend(time, 2)
plot_series(time[:200], series[:200])
plt.show()

series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
plot_series(time[:200], series[:200])
plt.show()

series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
series2 = autocorrelation(time, 5, seed=42) + seasonality(time, period=50, amplitude=2) + trend(time, -1) + 550
series[200:] = series2[200:]
# series += noise(time, 30)
plot_series(time[:300], series[:300])
plt.show()


def impulses(time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=10)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series


series = impulses(time, 10, seed=42)
plot_series(time, series)
plt.show()


def autocorrelation(source, φs):
    ar = source.copy()
    max_lag = len(φs)
    for step, value in enumerate(source):
        for lag, φ in φs.items():
            if step - lag > 0:
                ar[step] += φ * ar[step - lag]
    return ar


signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.99})
plot_series(time, series)
plt.plot(time, signal, "k-")
plt.show()

signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.70, 50: 0.2})
plot_series(time, series)
plt.plot(time, signal, "k-")
plt.show()

series_diff1 = series[1:] - series[:-1]
plot_series(time[1:], series_diff1)

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(series)

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

df = pd.read_csv("sunspots.csv", parse_dates=["Date"], index_col="Date")
series = df["Monthly Mean Total Sunspot Number"].asfreq("1M")
series.head()

series.plot(figsize=(12, 5))

series["1995-01-01":].plot()

series.diff(1).plot()
plt.axis([0, 100, -50, 50])

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(series)

autocorrelation_plot(series.diff(1)[1:])

autocorrelation_plot(series.diff(1)[1:].diff(11 * 12)[11 * 12 + 1:])
plt.axis([0, 500, -0.1, 0.1])

autocorrelation_plot(series.diff(1)[1:])
plt.axis([0, 50, -0.1, 0.1])

116.7 - 104.3

[series.autocorr(lag) for lag in range(1, 50)]

pd.read_csv(filepath_or_buffer, sep=',', delimiter=None,
            header='infer', names=None, index_col=None,
            usecols=None, squeeze=False, prefix=None,
            mangle_dupe_cols=True, dtype=None, engine=None,
            converters=None, true_values=None, false_values=None,
            skipinitialspace=False, skiprows=None, skipfooter=0,
            nrows=None, na_values=None, keep_default_na=True,
            na_filter=True, verbose=False, skip_blank_lines=True,
            parse_dates=False, infer_datetime_format=False,
            keep_date_col=False, date_parser=None, dayfirst=False,
            iterator=False, chunksize=None, compression='infer',
            thousands=None, decimal=b'.', lineterminator=None,
            quotechar='"', quoting=0, doublequote=True, escapechar=None,
            comment=None, encoding=None, dialect=None, tupleize_cols=None,
            error_bad_lines=True, warn_bad_lines=True, delim_whitespace=False,
            low_memory=True, memory_map=False, float_precision=None)

from pandas.plotting import autocorrelation_plot

series_diff = series
for lag in range(50):
    series_diff = series_diff[1:] - series_diff[:-1]

autocorrelation_plot(series_diff)

series_diff1 = pd.Series(series[1:] - series[:-1])
autocorrs = [series_diff1.autocorr(lag) for lag in range(1, 60)]
plt.plot(autocorrs)
plt.show()

##################
# Naive Forecast #
##################

naive_forecast = series[split_time - 1:-1]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150)  # zoom in
plot_series(time_valid, naive_forecast, start=1, end=151)

print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())


def moving_average_forecast(series, window_size):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)


moving_avg = moving_average_forecast(series, 30)[split_time - 30:]  # 30 points prior to it

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)

print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())

# seasonality 를 제외하고 raw 데이터를 보는 과정
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()

diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()

# 과거의 noisy data 를 더한 결과
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())

# 과거의 smooth data 를 더한 결과 - error rate 가장 improved
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
