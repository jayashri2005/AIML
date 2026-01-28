import numpy as np 
import matplotlib
from pydantic.type_adapter import R
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd

ar = np.array(['rat', 'cat', 'dog', 'cat', 'rat'])

from sklearn.preprocessing import  OneHotEncoder
ohe = OneHotEncoder()
ohe.fit(ar.reshape(-1,1))
ohe.transform(ar.reshape(-1,1))
sns.boxplot(ohe.transform(ar.reshape(-1,1)).toarray())
plt.savefig("onehot_boxplot.png")
plt.clf()

from sklearn.preprocessing import PowerTransformer

pt=PowerTransformer()
# Use numerical data for PowerTransformer, not categorical
numerical_data = np.random.normal(0, 1, 100).reshape(-1, 1)
trs=pt.fit(numerical_data)

df=pd.DataFrame({
    'experience':np.random.randint(10,40,50),
    'previous_salary':np.random.randint(1000,2000,50),
})

df.boxplot()
plt.savefig("databias.png")
plt.clf()

from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc = StandardScaler()
sttrans=sc.fit_transform(df)
pd.DataFrame(sttrans).boxplot()
plt.savefig("databias1.png")
plt.clf()

# MinMaxScaler example
mm = MinMaxScaler()
mmtrans=mm.fit_transform(df)
pd.DataFrame(mmtrans).boxplot()
plt.savefig("databias2.png")
plt.clf()

ar1=np.random.normal(30,0.5,30)
ar2=np.random.normal(200,0.5,30)
ar3=np.random.normal(400,0.5,30)

newar=np.stack([ar1,ar2,ar3],axis=1).flatten()
print(newar.shape)
sns.displot(newar,kind='kde')
plt.savefig("databias3.png")
plt.clf()

dfn=pd.DataFrame(newar)
sns.displot(dfn)
plt.savefig("databias4.png")
plt.clf()

print(newar[newar<80])
print(np.where(newar<80))
dfn.loc[dfn[0]<40,'new']='small'
print(dfn)
dfn.loc[(dfn[0]>150) & (dfn[0]<250),'new']='medium'
print(dfn)
dfn.loc[dfn[0]>350,'new']='large'
print(dfn)

def xyz(x):
    if x<40:
        return 'small'
    elif x>150 and x<250:
        return 'medium'
    else:
        return 'large'
dfn[0]=dfn[0].apply(xyz)
print(dfn)

from sklearn.preprocessing import KBinsDiscretizer

age=np.array([[18],[22],[29],[35],[42],[55],[68]])

kbd=KBinsDiscretizer(n_bins=5)
kbd.fit(age)
kbd.transform(age).toarray()
print(kbd.transform(age).toarray())

