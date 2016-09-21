
 # X1     X2      B       Y
 # 0      0       1       0
 # 0      1       1       1
 # 1      0       1       1
 # 1      1       1       0


# Apply Dropout only when NN is training

import numpy as np

X=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y=np.array([[0],[1],[1],[0]])

alpha=0.5
hidden_nodes=4
dropout_percent=0.2
do_dropout=True

np.random.seed(1)

wt0=2*np.random.random((3,hidden_nodes))-1
wt1=2*np.random.random((hidden_nodes,1))-1

for iter in range(70001):
    a0=X
    a1=1/(1+np.exp(-(np.dot(a0,wt0))))
    if(do_dropout):
        a1=a1*np.random.binomial([np.ones((len(a0),hidden_nodes))],1-dropout_percent)[0]*(1/(1-dropout_percent))
    a2=1/(1+np.exp(-(np.dot(a1,wt1))))
    if (iter % 10000) == 0:
        print "Error after " + str(iter) + " iterations:" + str(np.mean(np.abs(Y-a2)))
    a2_delta=(a2-Y)*(a2*(1-a2))
    a1_delta=a2_delta.dot(wt1.T)*(a1*(1-a1))
    wt1-=alpha*a1.T.dot(a2_delta)
    wt0-=alpha*a0.T.dot(a1_delta)

print " Output : "
print a2
