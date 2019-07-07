import numpy as np
data=[[0,1,1],[0,0,0],[1,0,1],[1,1,1],[1,1,1]]

def sigmoid(data):  
    return 1/(1+np.exp(-data))
def der_sigmoid(data):
    return sigmoid(data)*(1-sigmoid(data))
def train():
    w1=np.random.randn()
    w2=np.random.randn()
    b=np.random.randn()
    learning_rate=0.1
    for i in range(40000):
        j=np.random.randint(len(data))
        inputs = data[j]
        x=sigmoid(w1*inputs[0]+w2*inputs[1]+b)
        predicted=inputs[2]
        costfun=((x-predicted)**2)
        dcost_dpred = 2*costfun
        dpred_dz = der_sigmoid(x)
        dz_dw1=inputs[0]
        dz_dw2=inputs[1]
        dz_db=1
        dcost_dz=dcost_dpred*dpred_dz
        dw1=dcost_dz*dz_dw1
        dw2=dcost_dz*dz_dw2
        db=dcost_dz*dz_db
        w1=w1-learning_rate+dw1
        w2=w2-learning_rate+dw2
        b=b-learning_rate+db
    return w1,w2,b,costfun
w1, w2, b,cost = train()
a1=[1,0]
z = w1 * a1[0]+ w2 * a1[1]+ b
out=sigmoid(z)
if out<0.5:
    print("Football")
else:
    print("cricket")
    
    
