import numpy as np
import matplotlib.pyplot as plt

rnd = np.random.random((50,50))
print(rnd)
plt.matshow(rnd);
plt.colorbar()
plt.show()