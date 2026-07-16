# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt

print("pynapple version: ", nap.__version__)

# %%
# create a Ts object
spikes = nap.Ts(t=np.array([0.1, 0.3, 0.7, 1.2]))
print("spikes")
print(spikes)

# %%
# create a TsGroup object
spikes_1 = nap.Ts(t=np.array([0.1, 0.3, 0.7, 1.2]))
spikes_2 = nap.Ts(t=np.array([0.2, 0.4, 0.8, 1.3]))
spikes_group = nap.TsGroup({0: spikes_1, 1: spikes_2})
print("spikes_group")
print(spikes_group)


# %%
# create an IntervalSet object
intervals = nap.IntervalSet(start=np.array([0.1, 0.3, 0.7, 1.2]), end=np.array([0.2, 0.4, 0.8, 1.3]))
print("intervals")
print(intervals)

# %%
# create a time series (Tsd) and restrict it to epochs using intervalset
import pynapple as nap
import numpy as np

# create a time series
ts = nap.Tsd(t=np.arange(0, 6, 0.5), d=np.random.rand(12))
intervalset = nap.IntervalSet(start=[0, 4], end=[1, 5])
restricted_ts = ts.restrict(intervalset)
print("Original data:")
print(ts)
print("Restricted data:")
print(restricted_ts)
print("Restriction does not modify the original data:")
print(ts)


# %%
# use TsdFrame to create and plot several time series

time = np.linspace(0, 10, 100)
data = np.zeros((100, 3))
data[:,0] = np.sin(time) + np.random.normal(0, 0.1, 100)
data[:,1] = np.cos(time) + np.random.normal(0, 0.1, 100)
data[:,2] = np.tanh(time) + np.random.normal(0, 0.1, 100)
tsd = nap.TsdFrame(time, data)
plt.plot(tsd)
plt.title("TsdFrame example")
plt.show()

# %%
