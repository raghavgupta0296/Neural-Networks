   ##*      *#*
   #*#      #*#
   #*#      ###
   #*#      #*#
   ##*      #*#

   # D      # A

import numpy as np

D=np.array([[1,1,-1,1,-1,1,1,-1,1,1,-1,1,1,1,-1]])
A=np.array([[-1,1,-1,1,-1,1,1,1,1,1,-1,1,1,-1,1]])

inputPattern=np.array([[-1,1,-1,1,-1,1,0,-1,-1,1,-1,1,1,-1,1]])

yD=np.array([[-1,1,1,-1]])
yA=np.array([[-1,-1,1,1]])

wD=D.T*yD
wA=A.T*yA

W=wD+wA

x=inputPattern
y=np.array([[0,0,0,0]])

def activationFn(a_in):
    for n in range(0,4):
        if a_in[0][n]>0:
            a_in[0][n]=1
        elif a_in[0][n]<0:
            a_in[0][n]=-1
    return a_in


while(True):
    y_in=x.dot(W)
    y=activationFn(y_in)
    x_in=y.dot(W.T)
    x=activationFn(x_in)
    chkY=activationFn(x.dot(W))
    condition= chkY==y
    if(condition.all):
        print chkY
        break

