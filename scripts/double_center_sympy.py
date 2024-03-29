"""Symbol math for double centering the EDM.

The symbol math shows the transformation between the EDM and Gram
matrix in Multidimensional Scaling [1]_.

References
----------
.. [1] Dokmanic, R. Parhizkar, J. Ranieri, and M. Vetterli,
       “Euclidean Distance Matrices: Essential theory, algorithms, and
       applications,”IEEE Signal Processing Magazine, vol. 32, no. 6,
       pp. 12–30, nov 2015.

"""


from sympy import *

n = 4

D11, D12, D13, D14 = symbols("D11 D12 D13 D14")
D21, D22, D23, D24 = symbols("D21 D22 D13 D24")
D31, D32, D33, D34 = symbols("D31 D32 D13 D34")
D41, D42, D43, D44 = symbols("D41 D42 D13 D44")

J = eye(n) - (1/n)*ones(n,n)
D = Matrix([[0,   D12, D13, D14],
            [D21, 0,   D23, D24],
            [D31, D32, 0,   D34],
            [D41, D42, D43, 0]])
print(5*"\n")
G = -0.5*J*D*J
for ii in range(n):
    for jj in range(n):
        print(G[ii,jj],"\n")
