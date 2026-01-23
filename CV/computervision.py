from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import mglearn as mg
import matplotlib
matplotlib.use('Agg')

dt=load_digits(n_class=3)
dt.images[1]
y=dt.target
images=dt.images
print(y[-2])
print(images[0].shape)
images[0].reshape(1,64)
print(images[0].flatten())
print(images.shape)
x=images.reshape(-1,64)
print(x.shape)

