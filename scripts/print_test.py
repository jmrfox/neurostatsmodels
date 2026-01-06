import pynapple as nap
import numpy as np

t = np.arange(100)
d = np.ones((100, 30))
tsdframe = nap.TsdFrame(t=t, d=d)
print(tsdframe)
