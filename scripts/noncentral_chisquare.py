"""Plotting the noncentral chisquare distribution.

"""


import numpy as np
import matplotlib.pyplot as plt

sigma = 1
mu = 9.

plt.hist(sigma**2 * np.random.noncentral_chisquare(df = 1, nonc = (mu/sigma)**2, size = 10000),
         bins = 20, density = True)

plt.show()
