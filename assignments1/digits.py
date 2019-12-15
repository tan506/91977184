
"""
Created on Mon Dec  2 12:43:41 2019

@author: haruna
"""
import numpy as np
import pickle

class Digits(object):
    def __init__(self,path:str,filename:str):
        """
        initializes a Sonar instance
        path- relative path to data file
        filename- name of the file containing data
        """
        self.fullpath=path+filename 
        self.x,self.y=self.extract_data() 
        self.samples=self.x.shape[0] 
        self.y=self.y.astype(int)
        self.features=self.x.shape[1] 
        self.one_hot() 
        self.hidden=512 
        #initializing weights and biases
        self.w1=np.random.randn(self.features,self.hidden)
        self.b1=np.random.randn(self.hidden)
        self.w2=np.random.randn(self.hidden,10)
        self.b2=np.random.randn(10)
        self.learning_rate=0.0001 #setting learning rate
        
    def extract_data(self):
        """
        extracts the data stored in a csv file
        """
        file=open(self.fullpath,'rb')
        data=pickle.load(file)
        data=data["train"]
        num_el=[data[x].shape[0] for x in data]
        sumel=sum(num_el)
        temp=np.zeros((sumel,28*28+1))
        ind=0
        for num in data:
            temp[ind:ind+data[num].shape[0],0]=num
            temp[ind:ind+data[num].shape[0],1:]=data[num].reshape((data[num].shape[0],28*28))
            ind+=data[num].shape[0]
            
        data=temp
        file.close()
        y=data[:,0] 
        x=data[:,1:] 
        x=x/255 
        return (x,y)
    
    def forward_pass(self):
        """
        Forward passing the training set through the neural network using current weights and biases
        returns the outputs of the softmax and the sigmoid layers, which will be used in backpropagation
        """
        Z1=self.x.dot(self.w1)+self.b1
        A1=sigmoid(Z1)
        Z2=A1.dot(self.w2)+self.b2
        return softmax(Z2),A1
    
    def one_hot(self):
        """
        converts the training labels to one-hot representation
        """
        Y=np.zeros((self.samples,10))
        Y[np.arange(self.y.size),self.y]=1
        self.Y=Y
        
    def loss(self,y):
        """
        returns the current loss on the training set
        """
        return -(1/self.samples)*np.sum(np.sum(self.Y*np.log(y)))
    
    def train(self,epochs:int=100):
        """
        trains the neural network for the given number of epochs using back-propagation
        """
        print("Training Started")
        for i in range(epochs): 
            print("Epoch:",i)
            Y,A1=self.forward_pass() 
            print("Loss:",self.loss(Y))
            print("Accuracy:",self.accuracy())
            #calculating gradients
            d2=Y-self.Y
            d1=d2.dot(self.w2.T)*A1*(1-A1)
            #updating weights and bias of softmax layer
            self.w2-=self.learning_rate*A1.T.dot(d2)
            self.b2-=self.learning_rate*d2.sum(axis=0)
            #updating weights and bias of sigmoid layer
            self.w1-=self.learning_rate*self.x.T.dot(d1)
            self.b1-=self.learning_rate*d1.sum(axis=0)
        print("Final Loss:",self.loss(self.forward_pass()[0]))
        print("Final Accuracy:",self.accuracy())
    
    def accuracy(self):
        """
        returns the accuracy on the training set
        """
        Y=self.forward_pass()[0]
        y=np.argmax(Y,axis=1)
        return sum(y==self.y)/self.samples
    
    def predict(self,x):
        """
        predicts the label for the input feature vector or array x
        x should be a 2 dimensional array with rows as the samples and the columns as the features
        """
        Z1=x.dot(self.w1)+self.b1
        A1=sigmoid(Z1)
        Z2=A1.dot(self.w2)+self.b2
        if len(Z2.shape)==1:
            Z2=Z2.reshape((1,Z2.shape[0]))
        return np.argmax(softmax(Z2),axis=1)
     
    def save_model(self):
        """
        saves model into digit.pickle
        """
        filename="digit.pickle"
        file=open(filename,"wb")
        pickle.dump(self,file)
        file.close()
        
    @classmethod
    def load_model(cls):
        """
        class method
        returns the sonar model loaded from digit.pickle
        """
        filename="digit.pickle"
        file=open(filename,"rb")
        sonar_model=pickle.load(file)
        file.close()
        return sonar_model

def softmax(A):
    """
    returns the softmax function evaluated on A
    A should be a 2 dimensional array with the rows representing samples and columns representing the features
    """
    expA=np.exp(A)
    return expA/expA.sum(axis=1,keepdims=True)

def sigmoid(X):
    """
    returns the sigmoid function evaluated on X
    """
    return 1/(1+np.exp(-X))



path="/Users/haruna/Desktop/AI_1/assignment1/"
filename="digits_data.pkl"
digitmodel=Digits(path,filename)
digitmodel.train(25)
digitmodel.save_model()
model1=Digits.load_model()
print("Loaded Model Accuracy:",model1.accuracy())

