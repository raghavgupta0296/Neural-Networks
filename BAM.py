   #  Bidirectional Associative Net
   # Raghav Gupta
   # RA1411003010390
   # CSE 3rd Year

   ##*      *#*         ###
   #*#      #*#         #**
   #*#      ###         #**
   #*#      #*#         #**
   ##*      #*#         ###

   # D      # A         # C

import numpy as np

D=np.array([[1,1,-1,1,-1,1,1,-1,1,1,-1,1,1,1,-1]])
A=np.array([[-1,1,-1,1,-1,1,1,1,1,1,-1,1,1,-1,1]])
C=np.array([[1,1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,1]])

#Sample input Pattern -1,1,-1,1,-1,1,0,-1,-1,1,-1,1,1,-1,1 or -1,-1,1,1

inp = []

inputPattern = raw_input("Enter input pattern")
for i in inputPattern.split(','):
    inp.append(int(i))

inp = np.array(inp)

if inp.size == 15:
    print "Input Pattern : ",
    ctr=0
    for e in inp:
        if ctr%3==0:
            ctr=0
            print "\n"
        if e==-1:
            print "*",
            ctr+=1
        elif e==1:
            print "#",
            ctr += 1
        else:
            print " ",
            ctr += 1

inp = inp.reshape(1,inp.size)

yD=np.array([[-1,1,-1,1]])
yA=np.array([[1,-1,1,-1]])
yC=np.array([[-1,-1,1,1]])


wD=D.T*yD
wA=A.T*yA
wC=C.T*yC


W=wD+wA+wC

if inp.shape == (1,4):
    W=W.T

x=inp
y=np.array([[0,0,0,0]])

def activationFn(a_in):
    for n in range(0,a_in.size):
        if a_in[0][n]>0:
            a_in[0][n]=1
        elif a_in[0][n]<0:
            a_in[0][n]=-1
    return a_in

print "\n\n Weight Matrix : "
print W

print "\n Output Pattern : "

while(True):
    y_in=x.dot(W)
    y=activationFn(y_in)
    x_in=y.dot(W.T)
    x=activationFn(x_in)
    chkY=activationFn(x.dot(W))
    condition= chkY==y
    if(condition.all()):
        print chkY
        break


if inp.shape == (1,15):
    if (chkY==yD).all():
        print "\n Predicted : D"
    elif (chkY==yA).all():
        print "\n Predicted : A"
    elif (chkY==yC).all():
        print "\n Predicted : C"
    else:
        print "\n Not Similar To Trained Characters"

if inp.shape == (1,4):
    tD, tA, tC = 0, 0, 0
    for e in range(0,chkY.size):
        if chkY[0][e] != 0:
            if chkY[0][e] != D[0][e]:
                tD+=1
            if chkY[0][e] != A[0][e]:
                tA+=1
            if chkY[0][e] != C[0][e]:
                tC+=1
    if tD == 0:
        print "\n Predicted : D"
    if tA == 0:
        print "\n Predicted : A"
    if tC == 0:
        print "\n Predicted : C"
