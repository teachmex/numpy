## Meshgrid : Optimization with Gradient Descent


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```

#### Example 1: Simple Cost Function
$$z = x^2 + y^2$$


```python
ns = 10
xs = np.arange(-3,3,1/ns)
ys = np.arange(-3,3,1/ns)

XG, YG = np.meshgrid(xs, ys, sparse=True)  
ZG = XG**2 + YG**2
```

Update position in Gradient Descent

$$x \leftarrow x - \alpha \frac{\partial f(x,y)}{\partial x}$$

$$y \leftarrow y - \alpha \frac{\partial f(x,y)}{\partial y}$$


```python
alpha = 0.01
x,y = 2,2
tol = 1e-5
X,Y = [],[]
for i in range(1000):
    x = x - alpha*2*x
    y = y - alpha*2*y
    X.append(x)
    Y.append(y)
    if x*x+y*y < tol:
        print("at step", i," minimum found!",x,y,x*x+y*y)
        break
```

    at step 336  minimum found! 0.0022091108263220106 0.0022091108263220106 9.760341285946233e-06



```python
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure(figsize = [8,6])
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(XG, YG, ZG,\
                       cmap=cm.coolwarm,\
                       linewidth=0.5)
Z = [x**x+y*y for x,y in zip(X,Y)]
ax.scatter3D(X,Y,Z)
plt.show()
```


![png](output_6_0.png)



```python
plt.figure(figsize = [10,8])
CS = plt.contour(xs,ys,ZG)
plt.scatter(X,Y,marker="*")
plt.clabel(CS)
```




    <a list of 12 text.Text objects>




![png](output_7_1.png)


#### Example 2:  More Complex Cost Function

$$\large{Z = -\frac{sin(x^2+y^2).cos(x^2-y^2)}{x^2+y^2}}$$


```python
ns = 10
xs = np.arange(-3,3,1/ns)
ys = np.arange(-3,3,1/ns)

XG, YG = np.meshgrid(xs, ys, sparse=True)  
```


```python
ZG = -np.sin(XG**2 + YG**2)*np.cos(XG**2 - YG**2) / (XG**2 + YG**2)
```

- Update position in Gradient Descent

$$x \leftarrow x - \alpha \frac{\partial f(x,y)}{\partial x}$$

$$y \leftarrow y - \alpha \frac{\partial f(x,y)}{\partial y}$$

#### Optimization with Gradient Descend


```python
import sympy as sp
from sympy import diff, sin, cos,exp 
from sympy.abc import x,y 
```


```python
sp.diff(-sin(x*x+y*y)*cos(x*x-y*y)/(x*x+y*y),x)
```




    2*x*sin(x**2 - y**2)*sin(x**2 + y**2)/(x**2 + y**2) 
    - 2*x*cos(x**2 - y**2)*cos(x**2 + y**2)/(x**2 + y**2) 
    + 2*x*sin(x**2 + y**2)*cos(x**2 - y**2)/(x**2 + y**2)**2




```python
sp.diff(-sin(x*x+y*y)*sin(x*x-y*y)/(x*x+y*y),y)
```




    -2*y*sin(x**2 - y**2)*cos(x**2 + y**2)/(x**2 + y**2) 
    + 2*y*sin(x**2 + y**2)*cos(x**2 - y**2)/(x**2 + y**2) 
    + 2*y*sin(x**2 - y**2)*sin(x**2 + y**2)/(x**2 + y**2)**2




```python
from numpy import sin,cos
```


```python
def fun(x,y):
    f = -sin(x*x+y*y)*cos(x*x-y*y)/(x*x+y*y)
    return f

def find_diff(x,y):
    
    delx = 2*x*sin(x**2 - y**2)*sin(x**2 + y**2)/(x**2 + y**2) \
        + 2*x*cos(x**2 - y**2)*cos(x**2 + y**2)/(x**2 + y**2) \
        - 2*x*sin(x**2 + y**2)*cos(x**2 - y**2)/(x**2 + y**2)**2
    
    dely = -2*y*sin(x**2 - y**2)*sin(x**2 + y**2)/(x**2 + y**2) \
      + 2*y*cos(x**2 - y**2)*cos(x**2 + y**2)/(x**2 + y**2)\
        - 2*y*sin(x**2 + y**2)*cos(x**2 - y**2)/(x**2 + y**2)**2
    
    return delx,dely
```


```python
alpha = 0.01
x,y = 1,1
tol = 1e-5

'''There are many local minima, 
Gradient discend does not find global minimum'''
X,Y = [],[]
for i in range(10000):
    fdx,fdy = find_diff(x,y)
    x = x - alpha*fdx
    y = y - alpha*fdy
    X.append(x)
    Y.append(y)
```


```python
x,y,fun(x,y)
```




    (1.4989011738452018, 1.4989011738452018, 0.21723362821122166)




```python
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure(figsize = [10,8])
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(XG, YG, ZG,\
                       cmap=cm.coolwarm,\
                       linewidth=0.0,\
                       antialiased=False)

Z = [fun(x,y) for x,y in zip(X,Y)]
ax.scatter3D(X,Y,Z)
plt.show()
```


![png](output_21_0.png)



```python
plt.figure(figsize = [10,8])
CS = plt.contour(xs,ys,ZG)
plt.scatter(X,Y)
plt.clabel(CS)
```




    <a list of 36 text.Text objects>




![png](output_22_1.png)


------------

### Other Optimization Methods:
1. Method of Stepest Descend
2. Newton-Raphson Method
3. Newton-Raphson-Cartan Method
4. Coordinate Descent Method
5. Conjugate Gradient Method
6. Stochastic gradient method



