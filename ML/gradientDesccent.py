import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#df = pd.read_csv('grad_data.csv', index_col='Unnamed: 0')
x=np.linspace(3,8,12)
y=2*x+3+np.random.normal(0, 1, 12)
df=pd.DataFrame({'x':x,'y':y})
x = df['x'].values
y = df['y'].values
plt.scatter(x,y)
plt.savefig('scatter_plot.png')
plt.clf()

yp = 0*x + 0
plt.scatter(x,y)
plt.plot(x,yp,'*r:')
plt.savefig('plot1.png')
plt.clf()

w = 0
b = 0
yp = w*x + b
plt.scatter(x,y)
plt.plot(x,yp,'*r:')
plt.savefig('plot2.png')
plt.clf()

n = 11
mse = np.mean((y-yp)**2)
b=0
b = b + (0.05)*mse
yp = w*x + b
plt.scatter(x,y)
plt.plot(x,yp,'*r:')
plt.plot(x,w*x + 0, '*b:')
plt.savefig('plot3.png')
plt.clf()

mse = np.mean((y-yp)**2)
b = b + (0.05)*mse
yp = w*x+b
plt.scatter(x,y)
plt.plot(x,yp,'*r:')
plt.plot(x,w*x + 0, '*b:')
plt.savefig('plot4.png')
plt.savefig('final_plot.png')

w=0
b=0
plt.scatter(x,y)
for i in range(60):
    yp = w * x + b
    plt.plot(x, yp)
    mse = np.mean((y - yp)**2)
    # deriv_dw = (-2) * np.mean(x * (y - yp))
    # deriv_db = (-2) * np.mean(y - yp)
    # Update weights and intercept
    # w = w + 0.001 * (mse)
    b = b + 0.01 * (mse)
    yp = w*x + b
    plt.plot(x,yp)

plt.savefig('plot5.png')
plt.clf()

w=0
b=0
plt.scatter(x,y)
for i in range(60):
    yp = w * x + b
    plt.plot(x, yp)
    mse = np.mean((y - yp)**2)
    #deriv_dw = (-2) * np.mean(x * (y - yp))
    deriv_db = (-2) * np.mean(y - yp)  # [dl/db]
    #print("db:",deriv_db)
    # Update weights and intercept
    # w = w + 0.001 * (mse)
    # b = b + 0.01 * (mse)
    b = b - 0.1*deriv_db
    yp = w*x + b
    plt.plot(x,yp)
plt.savefig('plot6.png')
plt.clf()

w=0
b=0
plt.scatter(x,y)
for i in range(60):
    yp = w * x + b
    plt.plot(x, yp)
    mse = np.mean((y - yp)**2)
    deriv_dw = (-2) * np.mean(x * (y - yp))
    deriv_db = (-2) * np.mean(y - yp)
    # Update weights and intercept
    # w = w + 0.001 * (mse)
    # b = b + 0.01 * (mse)
    # b = b - 0.1*deriv_db
    w = w - 0.01*deriv_dw
    yp = w*x + b
    plt.plot(x,yp)
    # print(mse,w,b)
plt.savefig('plot7.png')
plt.clf()

w=0
b=0
plt.scatter(x,y)
for i in range(6):
    yp = w * x + b
    plt.plot(x, yp)
    mse = np.mean((y - yp)**2)
    deriv_dw = (-2) * np.mean(x * (y - yp))
    deriv_db = (-2) * np.mean(y - yp)
    # Update weights and intercept
    # w = w + 0.001 * (mse)
    # b = b + 0.01 * (mse)
    b = b - 0.01*deriv_db
    w = w - 0.01*deriv_dw
    yp = w*x + b
    plt.plot(x,yp)
    print(mse,w,b)
plt.savefig('plot8.png')
plt.clf()

""" vertical slope 'inf' horizonatal slope '0' for =>
 "yp=wx+b" error=yI=yP [actual-predicted] aprx eq = avg e^2 => sum(e^2/n)  
gradiant has minus inside
pytorch automatically cal derivative mech ->above ->

b=b+gradient(on the parabola the line that tpuch the curve)

"""

w = 0.0
b = 0.0
lr = 0.1  # learning rate -> trail method [hit & try/manually ]

plt.scatter(x, y)
for i in range(60):
    yp = w * x + b
    plt.plot(x, yp, alpha=0.5)
    
    mse = np.mean((y - yp)**2)
    deriv_dw = -2 * np.mean(x * (y - yp))  # Fixed: uncomment + correct sign
    deriv_db = -2 * np.mean(y - yp)
    
    w = w - lr * deriv_dw
    b = b - lr * deriv_db
    
    yp = w * x + b
    plt.plot(x, yp)
plt.savefig('plot9.png')
plt.show()