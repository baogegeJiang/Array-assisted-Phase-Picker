import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['font.size']=6

w=np.random.rand(1000)
plt.plot(w)
plt.text(500,2,'abc')
plt.show()