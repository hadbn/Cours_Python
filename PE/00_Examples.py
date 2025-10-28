# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
# ---

import numpy as np

# +
# %%timeit
n = 1000000
x = np.linspace(0, 2*np.pi, n)

# la bonne façon
np.sin(x)         # np.sin appliquée au tableau x
# -

import math

# +
# %%timeit
n = 1000000
x = np.linspace(0, 2*np.pi, n)

# la mauvaise façon
for e in x:             # une boucle for sur un tableau numpy
                        # c'est toujours une mauvaise idée
    math.sin(e)
# -

SLICING

import numpy as np
A=np.arange(1,31) * 2
B1, B2 = np.reshape(A,(2,5,3))
print(A)
print(B1)
print(B2)
print(B1[1,2])

tab = np.arange(120).reshape(2,3,4,5)
out = tab[:,0,1:-1,1:-1]
print(tab)
print(out)

a = np.arange(1,13).reshape(2,2,3)*2
print(a)
print(a.base)
#b = a[::-1,::-1,::-1]
b=np.flip(a)
print(b)
print(b.base)
print()
print(b.base is a)

# # IMAGES

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 2,2
from matplotlib.pyplot import imshow as show

img = np.zeros((91,91,3), dtype=np.uint8)
show(img)

# +

img[:,:,:]=255
show(img)
# -

img[:,:,0::2]=0
show(img)

print(img[0,0])
print(img[-1,-1])

img[::10, ::10]=[0,0,255]
show(img)

img[::10,:]=[0,0,255]
img[:,::10]=[0,0,255]
show(img)



img = plt.imread("data/les-mines.jpg")

print(img.flags.writeable)
import copy
img = copy.deepcopy(img)
print(img.flags.writeable)

show(img)

print(img.itemsize)
print(img.dtype)
print(img.min())
print(np.max(img))
show(img[:10,:10])

img = plt.imread("data/les-mines.jpg")
for i in [2,5,10,20]:
    show(img[::i,::i])
    plt.show()

fig, (axes) = plt.subplots(1,4,figsize = (rcParams['figure.figsize'][0]*4, rcParams['figure.figsize'][1]))
for i, value in enumerate([2,5,10,20]):
    axes[i].imshow(img[::value,::value])


def show_window(l,c):
    minx = (img.shape[1]-c)//2
    maxx = minx+c
    miny = (img.shape[0]-l)//2
    maxy=miny+l
    plt.imshow(img[miny:maxy,minx:maxx,:])
    plt.show()
show_window(10,20)
show_window(200,300)

plt.imshow(img[-1:,-1:,:])

# ## canaux rgb de l'image

img = plt.imread("data/les-mines.jpg")

r,v,b=np.split(img,3,-1)
#r,v,b = img[:,:,0], img[:,:,1], img[:,:,2]
plt.imshow(b, cmap='gray')

fig, (axes) = plt.subplots(1,3,figsize = (rcParams['figure.figsize'][0]*3, rcParams['figure.figsize'][1]))
for i in range(3):
    img_t = np.zeros(img.shape, dtype=np.uint8)
    img_t[:,:,i] = img[:,:,i]
    axes[i].imshow(img_t)

fig, (axes) = plt.subplots(1,3,figsize = (rcParams['figure.figsize'][0]*3, rcParams['figure.figsize'][1]))
colors=['Reds', 'Greens', 'Blues']
for i in range(3):
    axes[i].imshow(img[:,:,i],cmap=colors[i])

img = plt.imread("data/les-mines.jpg")
img = copy.deepcopy(img)
# img[-200:,-200:,:]=0
# show(img)
# plt.show()
# img[-200:,-200:]=[219,112,147]
# show(img)
# plt.show()
img[-200:,-200:,:]=255
img[-200::2,-200:,1:]=0
show(img)
plt.show()

show(img[-20:,-20:])


# # agregate

# le but est de ne pas utiliser de fct numpy
def unravel_index(index, shape):
    size = 1
    for item in shape:
        size *= item
    coords = []
    
    for dim_size in shape:
        size = size //dim_size
        coords.append(index // size)
        index = index % size
    
    return np.array(coords)



np.ones((3,5), dtype=np.uint8)

np.random.randint(0,256,(3,5), np.uint8)

l = ['un', 'deux', 'trois', 'cinq']
tab = np.array(l)
tab[0] = 'quatre'

tab

tab = np.ones((6))
tab

tab1 = np.reshape(tab, (3,2))
tab1

tab[0]=2
tab1

type(math.sin)

import numpy as np
A=np.arange(1,31) * 2
B = np.reshape(A,(2,5,3))
B

tab = np.arange(30)

tab

tab[::-1]

# +
a = np.arange(1,13).reshape(2,2,3)*2
print(a)
print(a.base)
# -


#b = a[::-1,::-1,::-1]
b=np.flip(a)
print(b)
print(b.base)
print()
print(b.base is a)

lines, cols = np.indices((4, 4))

cols

(cols+1)%2


# +

def block_checkers(n, k):
    Y,X = np.indices((n*k,n*k))
    Yk, Xk = (Y//k)%2, (X//k)%2
    return (((Xk == 0) & (Yk == 1)) | ((Xk == 1) & (Yk == 0))).astype(int)


# -

def stairs(n):
    Y,X = np.indices((2*n+1,2*n+1))
    out = (X+Y)*(X<n+1)*(Y<n+1)
    out += (X[:,::-1]+Y[::-1,:])*(X>=n+1)*(Y>=n+1)
    out += (X + Y[::-1,:])*(X<n+1)*(Y>=n+1)
    out += (X[:,::-1] + Y)*(X>=n+1)*(Y<n+1)
    return out
print(stairs(4))

n=4
k=3
Y,X = np.indices((n*k,n*k))

Y

((Y//k) + (X//k))%2

# # PANDAS

import pandas as pd
pd.options.display.precision = 2

df = pd.read_csv('data/titanic.csv')

df.describe()

df.describe(include='all')

df['Age']

df.head(8)

df = pd.read_csv('data/titanic.csv')
df = df.set_index('PassengerId')
df

df['Name'][552]

# +
df = pd.read_csv('data/titanic.csv')
df.index
# -


df = df.set_index('PassengerId')
df.index

# # EXO

df = pd.read_csv('data/petit-titanic.csv')
df.head(2)

df = pd.read_csv('data/petit-titanic.csv', sep=';')
df.head(2)

df = pd.read_csv('data/petit-titanic.csv', sep=';', header=None)
df.head(2)

df_ref = pd.read_csv('data/titanic.csv')
df = pd.read_csv('data/petit-titanic.csv', sep=';', names=df_ref.columns)
df.head(2)

columns = ['Identifiant', 'Survécu', 'Pclass',
           'Nom', 'Genre', 'Age', 'SibSp', 'Parch',
           'Ticket', 'Tarif', 'Cabine', 'Embarquement']
df.columns = columns
df.head(2)

# # MASQUES

df = pd.read_csv('data/titanic.csv')
girls = (df['Age'] < 12) & (df['Sex'] == 'female')
girls.sum()

df[girls]

children = df['Age'] < 12
children.value_counts()

children.sum()

df['Age'].isna()

df['Age'].isna().sum()

df.isna().sum()

df.isna().sum(axis=1)

# # Exo Agregation

df.isna().sum().sum()

import numpy as np
np.sum(df.isna().to_numpy())
df.isna().to_numpy().sum()

# # Exo type colonnes

import pandas as pd
df = pd.read_csv('data/titanic.csv')
cols = ['Survived', 'Pclass', 'Sex', 'Embarked']
for col in cols:
    print(col)
    print(df[col].unique())

dft=df[cols]
dft.dtypes

# # Exo stats

import pandas as pd
df = pd.read_csv('data/titanic.csv')

df.isna().sum().sum()

print(len(df.Pclass.unique()))
print(df.Pclass.value_counts())

df.Sex.value_counts()/len(df)

df.Sex.value_counts(normalize=True)

rate = ((df['Age'] >= 20) & (df['Age'] <= 40)).sum()/len(df)
print(rate)

df.Survived.describe()

(df.Survived == 1).mean()

for id_class in [1,2,3]:
    print(f"class_{id_class}")
    for sex in ['male', 'female']:
        print(sex)
        print(((df['Sex']==sex) & (df['Pclass']==id_class) & (df['Survived']==1)).sum() /
             ((df['Sex']==sex) & (df['Pclass']==id_class)).sum())

# # Copie

df = pd.read_csv('data/titanic.csv', index_col='PassengerId')
df2 = df.copy()
df2.loc[552, 'Age'] = 100
df2.head(1)

df.head(1)

# # Ajoute colonne

df['Deceased'] = 1 - df['Survived']
df.head(3)

# # Sélection multiple

# comme avec un tableau numpy,
# si on ne précise pas les colonnes
# on les obtient toutes
df.loc[552]

# on peut passer des listes à loc/iloc
# pour sélectionner explicitement
# plusieurs lignes / colonnesa
df.loc[[552, 832]]

df.loc[[552, 832], ['Name', 'Pclass']]

# à nouveau pour les indices de colonnes
# la colonne d'index ne compte pas
df.iloc[[0, -1], [2, 1]]

# +
# pour sélectionner plusieurs colonnes
# le plus simple c'est quand même cette forme
df[['Name', 'Pclass']]

df[(df['Pclass'] != 1) & (df['Age']>=70)]
# -

# mais bien sûr on peut aussi faire
df.loc[:, ['Name', 'Pclass']]
df[df['Sex']=='female']

# # Exo passageres femmes 1ere classe

# +
# le code
df = pd.read_csv('data/titanic.csv', index_col='PassengerId')

df_survived = (df['Survived'] == 1)
print(   df_survived.sum()/len(df)   )

( ((df['Sex'] == 'female') & (df['Survived'] == 1) & (df['Pclass'] == 1)).sum()
  /((df['Sex'] == 'female') & (df['Pclass'] == 1)).sum()   )
# -

# # group by

import pandas as pd
df = pd.read_csv('data/titanic.csv', index_col='PassengerId')
by_sex = df.groupby(by='Sex')
by_sex

by_sex_class = df.groupby(by=['Sex', 'Pclass'])

by_sex_class.size().index

pd.cut(df['Age'], bins=[0, 12, 19, 55, 100])

df['Age-class'] = pd.cut(
    df['Age'],
    bins=[0, 12, 19, 55, 100],
    labels=['child', ' young', 'adult', '55+'])
df.head()

df.groupby(['Age-class', 'Pclass'])['Survived'].mean()









dft = df.pivot_table(
    values='Survived',
    index='Pclass',
    columns='Sex',
)

dft.columns

dft







df2 = df.pivot_table(
    values=['Survived', 'Age'],
    index='Pclass',
    columns='Sex',
)
df2

df3 = df.pivot_table(
    values='Age',
    index='Pclass',
    columns=['Sex', 'Embarked'],
)
df3

df3.columns

df3.index

df4 = df.pivot_table(
    values=['Age', 'Survived'],
    index=['Pclass', 'Embarked'],
    columns='Sex',
)
df4

df = pd.read_csv('data/wine.csv')

df.head()
mag_min = df['magnesium'].min()
mag_max = df.magnesium.max()
mag_mean = df.magnesium.mean()

df['mag-cat']=pd.cut(df['magnesium'], bins=[mag_min-1,mag_mean,mag_max], labels=['mag-low', 'mag-high'])

df4 = df.pivot_table(
    values=['color-intensity', 'flavanoids', 'magnesium'],
    index=['cultivator'],
    columns='mag-cat',
)

df4





# # Filter groupby

df = pd.read_csv("data/titanic.csv")
gb = df.groupby(by=['Sex', 'Pclass'])
extract = gb.filter(lambda df: len(df) %2 == 0)
extract

# # TD 2-07

len(df['Sex'].unique()) * len(df['Pclass'].unique()) * len(df['Survived'].unique())

groups = df.groupby(['Pclass', 'Sex', 'Survived'])
groups.size()

groups.get_group((1, 'female', 0))

mask = (df.Pclass == 1) & (df.Sex == 'female') & ~df.Survived
df[mask]

D = df.pivot_table(
    index=['Sex', 'Pclass'], values='Survived', aggfunc="mean")['Survived'].to_dict()
D

# +
D = {}
for group, subdf in df.groupby(['Sex', 'Pclass']):
    D[group] = subdf.Survived.sum() / len(subdf)

D
# -

pd.Series(D, name="taux de survie par genre dans chaque classe")

# https://numerique-exos.info-mines.paris/pandas-tps/leases/readme-leases-nb/

# # Matplotlib

# +
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,2*np.pi,100)
y1 = np.sin(x) #coords de la première courbe
y2 = np.cos(x) #coords de la seconde courbe
plt.plot(x, y1, label='sin', marker='o', color='red')
plt.plot(x, y2, label='cos', linestyle='dashed')
plt.xlabel('axe des X')
plt.ylabel('axe des Y')
plt.legend()
plt.title('Trigo')
plt.xlim(0, 2.1*np.pi)
plt.yticks([-1, 0 ,1])
plt.show()

# +

x = np.linspace(0,2*np.pi,100)
y1 = np.sin(x) #coords de la première courbe
y2 = 100 * np.cos(x)

ax1 = plt.subplot(2,2,1)
ax1.plot(x,y1, color='black', label='courbe')
ax1.plot(x, np.cos(x), color='red', linestyle='dashed', label='courbe 2')
ax1.set_xlim(-np.pi, 3*np.pi)
ax1.set_ylim(-1,1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.legend()
ax1.scatter([0,2,4,6], [0,1/4,1/2,3/4], color='blue', marker='v')


x = np.linspace(-2,2,100)
y1 = 2*np.pow(x,3) -5*x +3 #coords de la première courbe
y2 = 3*np.pow(x,2)
ax2 = plt.subplot(2,2,(3,4))
ax2.plot(x, y1)
ax2.plot(x,y2)
ax2.fill_between(x, y1, y2, hatch='/', alpha=0.1, color='red')
ax2.set_title('dernier graphique')

ax3 = plt.subplot(2,2,2, projection='3d')
ax3.scatter(np.random.random(20), np.random.random(20), np.random.random(20))
ax3.set_title('un graph en 3d')
plt.show()
