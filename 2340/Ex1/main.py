##################################################
# Author: Asaf Harel
##################################################

import matplotlib.pyplot as plt
import numpy as np

import unit10.b_utils as u10

import random

random.seed(1)

x, y = u10.load_dataB1W3Ex1()

plt.plot(x, y, 'b.')
plt.show()
