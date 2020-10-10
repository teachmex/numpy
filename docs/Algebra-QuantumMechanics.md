### Algebra: Quantum Mechanics

Reference: http://qutip.org/docs/latest/index.html


```python
import numpy as np
from qutip import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.linalg as LA
from scipy.linalg import expm
%matplotlib inline
sns.set()
```


```python
#!pip install qutip
```

### Some Facts from Quantum Physics:

1. Web function \\(\psi(x)\\) is projection of abstract quantum state \\(|\psi \rangle \\) (in a certain representation) to a position space (representation) basis \\(|x \rangle\\). Where, \\(|x\rangle\\) is a continuous basis with orthogonality relation $\langle x^{'}|x\rangle = \delta(x,x^{'})$.

$$ \psi(x) = \langle x|\psi \rangle$$

2. Web function \\(\psi(p)\\) is projection of abstract quantum state \\(|\psi \rangle \\) in a certain representation to a momentum space (representation)  basis \\(|p \rangle\\). Where, \\(|p\rangle\\) is a continuous basis with orthogonality relation \\(\langle p^{'}|p\rangle = \delta(p,p^{'})\\).

$$\phi(p) = \langle p|\phi \rangle$$



3. Spherical Harmonics \\(Y_{l,m}(\theta,\phi)\\) is projection of abstract quantum state \\(| l,m \rangle\\) (in angular momnetum representation) to a position space (representation) basis \\(|\theta, \phi \rangle\\).

$$Y_{l,m}(\theta,\phi) = \langle \theta,\phi|l,m \rangle$$

4. An unitary operator U can be constructed from exponentiation of Hermitian Operator H. 
$$U = exp(-i\alpha H)$$
One application of this approach is defining Rotation matrix R in Hilbert space by implementation of angular momentum Operator (e.g., \\(L_x, L_y, L_z\\)) as generator of rotation in specific irreducible subspace (e.g.,l=0,l=1,l=2...) of the Hilbert space. A general rotation in Hilbert space is infinite dimentional rotation matrix. In eigen basis of anfgular momentum (\\(L^{2}, L_z\\)), this matrix appears as block diagonal matrix with block representing rotation in specific irriducible sub-space.


5. In hydrogen like system with spinless particle, operator hamiltonian $H$, square of Angular momentum $L^{2}$ and z-omponent of $L$ operator(\\(L_z\\)) commute with eachother which means these operators are simultaneously diagonalized. The benefit of of this relation is that once we are able to find eigen basis of \\(L_z\\) operator, we get the eigen basis of Hamiltonian as well. The eighen states of hamiltonian represents the energy level of the system.

### 1. Hydrogen Atom: Angular Momentum and Spherical Harmonics

The matrix element of general angular momnetum operators $J,J_z,J_+,J_-$ are as follows:

![img](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTRMh9_FL09_N_ftKHwlFvsrwyJQv_wgu_hVg&usqp=CAU)

One can write a python function to provide a matrix element for an arbitrary operators $J,J_z,J_+,J_-$ , but we dont have to work hard now python package ```qutip``` provides us these operators as quantum object ```Quobj``` (of kind operators). We will try to play around with some of them.

----------

####   Spin Angular Momentum \\(s = 1/2, ms = -1/2, 1/2\\)
Matrix size = 2x2


```python
sigmax()
```




Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True\begin{equation*}\left(\begin{array}{*{11}c}0.0 & 1.0\\1.0 & 0.0\\\end{array}\right)\end{equation*}




```python
sx = np.array(sigmax())
sy = np.array(sigmay())
sz = np.array(sigmaz())
sx,sy,sz
```




    (array([[0.+0.j, 1.+0.j],
            [1.+0.j, 0.+0.j]]), array([[0.+0.j, 0.-1.j],
            [0.+1.j, 0.+0.j]]), array([[ 1.+0.j,  0.+0.j],
            [ 0.+0.j, -1.+0.j]]))




```python
np.dot(sx,sx), np.dot(sy,sy), np.dot(sz,sz)
```




    (array([[1.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]), array([[1.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]), array([[1.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j]]))



-----------

#### Experiment 1 :

- A generic rotation \\( U_n(\theta) = e^{-i \theta n.\sigma} = e^{-i (\theta_x \sigma_x + \theta_y \sigma_y + \theta_z \sigma_z)} \\)

Note: \\( U_n(\theta) = e^{-i \theta n.\sigma} \neq e^{-i (\theta_x \sigma_x)} e^{-i (\theta_y \sigma_y)} e^{-i(\theta_z \sigma_z)} \\) since pauli matrices are non-commuting.

- Generic quantum state in ($s =1/2$) subspace: \\(|\psi \rangle = \alpha |\psi_{1/2}\rangle + \beta |\psi_{-1/2}\rangle\\)

[Non-commuting Matrix exponentials](https://math.stackexchange.com/questions/588007/non-commuting-matrix-exponentials/590120)

---------

Let us tast above fact by evaluating generic rotation operator $U_n$ around a arbitrary axix $n$ (i.e., $e^{-i (\theta_x \sigma_x + \theta_y \sigma_y + \theta_z \sigma_z)})$ as ```U_direct``` and the product of individual rotation operator (i.e.,  $ e^{-i (\theta_x \sigma_x)} e^{-i (\theta_y \sigma_y)} e^{-i(\theta_z \sigma_z)}$) as ```U_prod``` in the code cells below. We will clearly see that thes two terms are not equal verifying relation $e^{-i \theta n.\sigma} \neq e^{-i (\theta_x \sigma_x)} e^{-i (\theta_y \sigma_y)} e^{-i(\theta_z \sigma_z)}$.

- Calculate \\(e^{-i (\theta_x \sigma_x + \theta_y \sigma_y + \theta_z \sigma_z)})\\) as ```U_direct```


```python
U_direct = expm(-1j*((np.pi/12)*sx + (np.pi/12)*sy + (np.pi/12)*sz))
U_direct
```




    array([[ 0.89894119-0.25291945j, -0.25291945-0.25291945j],
           [ 0.25291945-0.25291945j,  0.89894119+0.25291945j]])



- Calculate \\(e^{-i (\theta_x \sigma_x)} e^{-i (\theta_y \sigma_y)} e^{-i(\theta_z \sigma_z)}\\) as product of three individual rotation as ```U_prod```


```python
Ux = expm(-1j*np.pi/12*sx)
Uy = expm(-1j*np.pi/10*sy)
Uz = expm(-1j*np.pi/8*sz)
Uz,Ux,Uy
```




    (array([[0.92387953-0.38268343j, 0.        +0.j        ],
            [0.        +0.j        , 0.92387953+0.38268343j]]),
     array([[0.96592583+0.j        , 0.        -0.25881905j],
            [0.        -0.25881905j, 0.96592583+0.j        ]]),
     array([[ 0.95105652+0.j, -0.30901699+0.j],
            [ 0.30901699+0.j,  0.95105652+0.j]]))




```python
U_prod = np.dot(Ux,np.dot(Uy,Uz))
U_prod
```




    array([[ 0.81811516-0.42544356j, -0.18156837-0.34164059j],
           [ 0.18156837-0.34164059j,  0.81811516+0.42544356j]])



- Are ```U_direct``` and ```U_prod``` same Operators?
Why? 

We can see these two operators are not same by implement them in same initial state ```psi0``` vector and observe the final states are not same.


```python
psi0 = 1/np.sqrt(2)*np.array([1,1])
psi0
```




    array([0.70710678, 0.70710678])




```python
np.dot(U_direct,psi0), np.dot(U_prod,psi0)
```




    (array([0.45680635-3.57682117e-01j, 0.81448847+8.32667268e-17j]),
     array([0.45010655-0.5424104j , 0.706883  +0.05925765j]))



- In fact, both of them are Unitary operators with determinant 1


```python
LA.det(U_direct), LA.det(U_prod)
```




    ((0.9999999999999999+1.1102230246251564e-16j), (1.0000000000000002+0j))



####  Angular Momentum \\(l =1, m = -1,0,1\\)
Matrix size  = 3x3


```python
jmat(1)
```




    (Quantum object: dims = [[3], [3]], shape = (3, 3), type = oper, isherm = True
     Qobj data =
     [[0.         0.70710678 0.        ]
      [0.70710678 0.         0.70710678]
      [0.         0.70710678 0.        ]],
     Quantum object: dims = [[3], [3]], shape = (3, 3), type = oper, isherm = True
     Qobj data =
     [[0.+0.j         0.-0.70710678j 0.+0.j        ]
      [0.+0.70710678j 0.+0.j         0.-0.70710678j]
      [0.+0.j         0.+0.70710678j 0.+0.j        ]],
     Quantum object: dims = [[3], [3]], shape = (3, 3), type = oper, isherm = True
     Qobj data =
     [[ 1.  0.  0.]
      [ 0.  0.  0.]
      [ 0.  0. -1.]])




```python
LX = np.array(jmat(1,'x'))
LY = np.array(jmat(1,'y'))
LZ = np.array(jmat(1,'z'))
```

- Do \\(L_x, L_y\\) commute?


```python
np.dot(LX,LY) == np.dot(LY,LX)
```




    array([[False,  True,  True],
           [ True, False,  True],
           [ True,  True, False]])



- What is matrix element of $L^{2}$ ?


```python
L_square = (np.dot(LX,LX) + np.dot(LY,LY) +np.dot(LZ,LZ))
L_square
```




    array([[2.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 2.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 2.+0.j]])



-------

#### Experiment 2

- Rotaion \\( R(\theta) = e^{-i \theta n.L} = e^{-i (\theta_x L_x + \theta_y L_y + \theta_z L_z)} \\)

- Generic quantum state in ($l =0$) subspace: \\(|\psi \rangle = \alpha |\psi_{10}\rangle + \beta |\psi_{11} \rangle + \gamma |\psi_{1-1} \rangle\\)

Let us find rotation matrix for subspace (l=1) with different values of \\(\theta_x, \theta_y, \theta_z\\)


```python
Rx = expm(-(1.0j)*0.1*LX)
Ry = expm(-(1.0j)*0.2*LY)
Rz = expm(-(1.0j)*0.3*LZ)
R_prod = np.dot(Rx,np.dot(Ry,Rz))
R_direct = expm(-(1.0j)*(0.1*LX + 0.2*LY + 0.3*LZ))
```


```python
R_prod, R_direct
```




    (array([[ 0.94049792-3.01310652e-01j, -0.14048043-6.91857281e-02j,
              0.00420456+1.16811741e-02j],
            [ 0.11267399-1.08747365e-01j,  0.97517033+6.69983947e-18j,
             -0.11267399-1.08747365e-01j],
            [ 0.00420456-1.16811741e-02j,  0.14048043-6.91857281e-02j,
              0.94049792+3.01310652e-01j]]),
     array([[ 0.94316771-2.93048837e-01j, -0.14862798-4.81054052e-02j,
              0.00741291+9.88387642e-03j],
            [ 0.12766111-9.00391414e-02j,  0.97529031-1.50304582e-18j,
             -0.12766111-9.00391414e-02j],
            [ 0.00741291-9.88387642e-03j,  0.14862798-4.81054052e-02j,
              0.94316771+2.93048837e-01j]]))



**Mini Assignment:**
- Roate a random vector  \\(|\psi \rangle \\),i.e. (\\(|\psi \rangle = \alpha |\psi_{10}\rangle + \beta |\psi_{11} \rangle + \gamma |\psi_{1-1} \rangle)\\) by implementing ```R_prod``` and ``` R_direct``` calculated above and compere the final state vectors.


```python

```

#### Angular Momentum plus Spin: \\(l = 3/2, m = -3/2,-1/2,1/2,3/2\\)
Matrix size = 4x4


```python
jmat(3/2,'x')
```




Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = True\begin{equation*}\left(\begin{array}{*{11}c}0.0 & 0.866 & 0.0 & 0.0\\0.866 & 0.0 & 1.0 & 0.0\\0.0 & 1.0 & 0.0 & 0.866\\0.0 & 0.0 & 0.866 & 0.0\\\end{array}\right)\end{equation*}




```python
jmat(3/2,'y')
```




Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = True\begin{equation*}\left(\begin{array}{*{11}c}0.0 & -0.866j & 0.0 & 0.0\\0.866j & 0.0 & -1.0j & 0.0\\0.0 & 1.0j & 0.0 & -0.866j\\0.0 & 0.0 & 0.866j & 0.0\\\end{array}\right)\end{equation*}



### Summary: Rotation in Hilbert Space 

Structure of a general [rotation matrix (R) in Hilbert space](https://www2.ph.ed.ac.uk/~ldeldebb/docs/QM2/chap4.pdf)

j = 1/2,1,3/2,2,5/2,3,...

- A general rotation matrix ($R$) in hilbert space of basis \\(|l,m\rangle\\) appears as block diagonal matrix, wher every block represents the rotation with specific subspaces called irriducible subspace.
- In the same basis $|l,m\rangle$  Hamiltonian Matrix $H$, $J^2$ and $J_z$ are simultaneously diagonalized.

![img](https://physicscourses.colorado.edu/phys5250/phys5250_fa19/images/block-diag.png)

-----------

### 2. Quantum Harmonic Oscillator


```python
momentum(5)
```




Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True\begin{equation*}\left(\begin{array}{*{11}c}0.0 & -0.707j & 0.0 & 0.0 & 0.0\\0.707j & 0.0 & -1.0j & 0.0 & 0.0\\0.0 & 1.0j & 0.0 & -1.225j & 0.0\\0.0 & 0.0 & 1.225j & 0.0 & -1.414j\\0.0 & 0.0 & 0.0 & 1.414j & 0.0\\\end{array}\right)\end{equation*}




```python
position(5)
```




Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = True\begin{equation*}\left(\begin{array}{*{11}c}0.0 & 0.707 & 0.0 & 0.0 & 0.0\\0.707 & 0.0 & 1.0 & 0.0 & 0.0\\0.0 & 1.0 & 0.0 & 1.225 & 0.0\\0.0 & 0.0 & 1.225 & 0.0 & 1.414\\0.0 & 0.0 & 0.0 & 1.414 & 0.0\\\end{array}\right)\end{equation*}




```python
create(4)
```




Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = False\begin{equation*}\left(\begin{array}{*{11}c}0.0 & 0.0 & 0.0 & 0.0\\1.0 & 0.0 & 0.0 & 0.0\\0.0 & 1.414 & 0.0 & 0.0\\0.0 & 0.0 & 1.732 & 0.0\\\end{array}\right)\end{equation*}




```python
destroy(5)
```




Quantum object: dims = [[5], [5]], shape = (5, 5), type = oper, isherm = False\begin{equation*}\left(\begin{array}{*{11}c}0.0 & 1.0 & 0.0 & 0.0 & 0.0\\0.0 & 0.0 & 1.414 & 0.0 & 0.0\\0.0 & 0.0 & 0.0 & 1.732 & 0.0\\0.0 & 0.0 & 0.0 & 0.0 & 2.0\\0.0 & 0.0 & 0.0 & 0.0 & 0.0\\\end{array}\right)\end{equation*}




```python
num(4)
```




Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = True\begin{equation*}\left(\begin{array}{*{11}c}0.0 & 0.0 & 0.0 & 0.0\\0.0 & 1.0 & 0.0 & 0.0\\0.0 & 0.0 & 2.0 & 0.0\\0.0 & 0.0 & 0.0 & 3.0\\\end{array}\right)\end{equation*}



### 3. Random Matrices

- Random hermitian matrix


```python
rand_herm(4)
```




Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = True\begin{equation*}\left(\begin{array}{*{11}c}-0.501 & (-0.172+0.146j) & 0.0 & (-0.060+0.697j)\\(-0.172-0.146j) & 0.169 & (-0.305+0.588j) & (0.044+0.641j)\\0.0 & (-0.305-0.588j) & 0.0 & (-0.458-0.087j)\\(-0.060-0.697j) & (0.044-0.641j) & (-0.458+0.087j) & 0.756\\\end{array}\right)\end{equation*}



- Random Unitary Matrix


```python
rand_unitary(4)
```




Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = False\begin{equation*}\left(\begin{array}{*{11}c}(0.332-0.610j) & (-0.248+0.007j) & (-0.332-0.267j) & (0.468-0.235j)\\(-0.062-0.085j) & (0.838-0.353j) & (-0.312-0.253j) & (3.589\times10^{-05}-0.034j)\\(-0.032-0.481j) & (0.145-0.293j) & (0.790+0.106j) & (0.020-0.156j)\\(-0.427+0.302j) & (0.069+0.006j) & (0.109+0.097j) & (0.834-0.066j)\\\end{array}\right)\end{equation*}



### Mini Assignment:
1. Generate a random Hermitian matrix $H$ of size 10 by 10.
3. Diagonalize the Hermitian Operator $H$ and find eigne values and eigen vectors.
2. Create a Unitary operator $U$ by exponentiating the Hermitian matrix i.e. $U = e^{-i \alpha H}$.
3. Apply operator $U$ over eigen vectors of operator $H$. Does this operation preserve the norm of eigen vectors?
3. Check that $U$ is unitary or not.
4. What is the determinant of $U$?

### Reference
1. [Rotation](https://www2.ph.ed.ac.uk/~ldeldebb/docs/QM2/chap4.pdf)
2. [Spherical Harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics)
3. [Angular Momentum](https://theory.physics.manchester.ac.uk/~xian/qm/chapter2.pdf)
