
 # X1     X2      B       Y
 # 0      0       1       0
 # 0      1       1       1
 # 1      0       1       1
 # 1      1       1       0

import numpy as np

X=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y=np.array([[0],[0],[0],[1]])

np.random.seed(1)

wt0=2*np.random.random((3,4))-1
wt1=2*np.random.random((4,1))-1

for iter in range(1000):
    a0=X
    a1=1/(1+np.exp(-(np.dot(a0,wt0))))
    a2=1/(1+np.exp(-(np.dot(a1,wt1))))
    a2_delta=(Y-a2)*(a2*(1-a2))
    a1_delta=a2_delta.dot(wt1.T)*(a1*(1-a1))
    wt1+=a1.T.dot(a2_delta)
    wt0+=a0.T.dot(a1_delta)

print " Output : "
print a2

print " X "
print X
print " Y "
print Y
