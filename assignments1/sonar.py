import numpy as np
import pickle
class Sonar(object):
    
    def __init__(self,path:str,filename:str):
        """
        initializes a Sonar instance
        path- relative path to data file
        filename- name of the file containing data
        """
        
        file=open(path+filename,'rb')
        sonardata=pickle.load(file)
        file.close()
        self.data=sonardata
        self.datalen=self.data['m'].shape[1]
        self.num_samples=self.data['m'].shape[0]+self.data['r'].shape[0]
        self.weights=np.random.randn(self.datalen)
        self.learning_rate=0.1
        
            
    def train(self,epochs:int):
        """
        trains the sonar model
        epochs- number of epochs to be trained for
        """
        
        print("Initializing...")
        #getting the data ready
        #samples will be a list of lists. The elements of samples will be a list of two elements
        #first element is the numpy array of sonar vector
        #second element is the label +1 for mine, -1 for rock
        samples=[]
        for i in range(self.data['m'].shape[0]):
            samples.append([self.data['m'][i,:],1])
        for i in range(self.data['r'].shape[0]):
            samples.append([self.data['r'][i,:],-1])
        np.random.shuffle(samples)
        print("Training Started...")
        
        for epoch in range(epochs): 
            print("Epoch "+str(epoch+1))
            correct=0 
            losses=[] 
            for sample in samples:
                y_actual=sample[1]
                y_predict=self.predict(sample[0])
                loss=max(0,y_actual-np.dot(self.weights,sample[0]))    
                losses.append(loss)
                if y_actual==y_predict: 
                    correct+=1
                gradients=(y_actual-y_predict)*sample[0] 
                self.weights=self.weights+self.learning_rate*gradients 
            #printing accuracy and average loss for a particular epoch    
            print("Accuracy="+str(correct/self.num_samples))
            print("Average Loss="+str(sum(losses)/self.num_samples))
            
    def predict(self,v:np.array):
        """
        predicts the label for given sonar vector using the current weights
        v- sonar vector
        """
        a=np.dot(self.weights,v)
        y=0
        if a>=0:
            y=1
        else:
            y=-1
        return y   
    
    def save_model(self):
        """
        saves model into sonar.pickle
        """
        filename="sonar.pickle"
        file=open(filename,"wb")
        pickle.dump(self,file)
        file.close()
        
    @classmethod
    def load_model(cls):
        """
        class method
        returns the sonar model loaded from sonar.pickle
        """
        filename="sonar.pickle"
        file=open(filename,"rb")
        sonar_model=pickle.load(file)
        file.close()
        return sonar_model
        
path="/Users/haruna/Desktop/AI_1/assignment1/"
filename="sonar_data.pkl"
sonar_model=Sonar(path,filename)
sonar_model.train(100)
sonar_model.save_model()
v=sonar_model.data['m'][67]
print("Model1 prediction:",sonar_model.predict(v))
sonar_model1=Sonar.load_model()
print("Loaded Model prediction:",sonar_model1.predict(v))
                

