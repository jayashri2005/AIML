import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mglearn as mg
from sklearn.linear_model import make_moons

x,y=make_moons(noise=0.15,random_state=21)
print(x.shape)
mg.discrete_scatter(x[:,0],x[:,1],y)
plt.legend
plt.show()
plt.savefig('regression1.png')