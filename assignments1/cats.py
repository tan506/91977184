################################################################################
#                                                                              #
#                               INTRODUCTION                                   #
#                                                                              #
################################################################################

# In order to help you with the first assignment, this file provides a general
# outline of your program. You will implement the details of various pieces of
# Python code grouped in functions. Those functions are called within the main
# function, at the end of this source file. Please refer to the lecture slides
# for the background behind this assignment.
# You will submit three python files (sonar.py, cat.py, digits.py) and three
# pickle files (sonar_model.pkl, cat_model.pkl, digits_model.pkl) which contain
# trained models for each tasks.
# Good luck!

################################################################################
#                                                                              #
#                                    CODE                                      #
#                                                                              #
################################################################################

import numpy as np
import pickle as pkl
import random

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def perceptron(z):
    return -1 if z<=0 else 1

# loss functions

def ploss(yhat, y):
    return max(0, -yhat*y)

def lrloss(yhat, y):
    return 0.0 if yhat==y else -1.0*(y*np.log(yhat)+(1-y)*np.log(1-yhat))
# prediction functions

def ppredict(self, x):
    return self(x)

def lrpredict(self, x):
    return 1 if self(x)>0.5 else 0



class Cat_Model:


    def __init__(self, dimension=12288, weights=None, bias=None, activation=(lambda x: x), predict=(lambda x: x)):
    
        self._dim = dimension
        self.w = weights or np.random.normal(size=self._dim)
        self.w = np.array(self.w)
        self.b = bias if bias is not None else np.random.normal()
        self._a = activation
        self.predict = predict.__get__(self)
    
    def __str__(self):
        
        return "Simple cell neuron\n\
        \tInput dimension: %d\n\
        \tBias: %f\n\
        \tWeights: %s\n\
        \tActivation: %s" % (self._dim, self.b, self.w, self._a.__name__)
    
    def __call__(self, x):
        
        yhat = self._a(np.dot(self.w, np.array(x)) + self.b)
        return yhat

    def load_model(self, file_path):
        '''
        open the pickle file and update the model's parameters
        '''
        with open(file_path,mode='rb') as f:
            mm = pkl.load(f)
        
        self._dim = mm._dim
        self.w = mm.w
        self.b = mm.b
        self._a = mm._a
        

    def save_model(self):
        '''
        save your model as 'cat_model.pkl' in the local path
        '''
        
        f = open ("cat_model.pkl",'wb')
        pkl.dump(self,f)
        f.close
        
class Cat_Trainer:

    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model
        self.loss = lrloss

    def accuracy(self, data):
        '''
        return the accuracy on data given data iterator
        '''
        return 100*np.mean([1 if self.model.predict(x) == y else 0 for x, y in data])
    

    def train(self, lr, ne):
        '''
        This method should:
        1. display initial accuracy on the training data loaded in the constructor
        2. update parameters of the model instance in a loop for ne epochs using lr learning rate
        3. display final accuracy
        '''
        print (lr)
        print("training model on data...")
        accuracy = self.accuracy(self.dataset)
        print("initial accuracy: %.3f" % (accuracy))
        
        costs = []
        accuracies =[]
        
        for epoch in range(1,ne+1):
            
            self.dataset.shuffle()
            J = 0
            dw = 0
            for d in self.dataset.samples:
                xi, yi = d
                yhat = self.model(xi)
                J += self.loss (yhat,yi)
                dz = yhat - yi
                dw += xi*dz
            J /= len(self.dataset.samples)
            dw /= len(self.dataset.samples)
            self.model.w = self.model.w - lr*dw
            
            accuracy = self.accuracy(self.dataset)
            
            if epoch % 10 == 0:
                print ('>epoch=%d,  accuracy=%.3f' % (epoch+1,lr, accuracy))
            costs.append(J)
            accuracies.append(accuracy)
            
            
            
        print("training complete")
        print("final accuracy: %.3f" % (self.accuracy(self.dataset)))
        costs = list(map(lambda t: np.mean(t),[np.array(costs)[i-10:i+11]for i in range (1, len(costs)+10)]))
        accuracies = list(map(lambda t: np.mean(t),[np.array(accuracies)[i-10:i+11]for i in range (1, len(accuracies)+10)]))
        
        return (costs, accuracies)

    
class Cat_Data:

    def __init__(self, relative_path='/Users/haruna/Desktop/AI_1/assignment1/', data_file_name='cat_data.pkl'):
        '''
        initialize self.index; load and preprocess data; shuffle the iterator
        '''
        self.index = -1
       
        with open ('%s%s'%(relative_path, data_file_name), mode='rb') as f:
            cat_data = pkl.load(f)
        self.samples = [(np.resharpe(vector, vector.size), 1) for vector in cat_data ['train']['cat']]+[(np.resharpe(vector, vector.size), 0) for vector in cat_data ['train']['cat']]
        self.shuffle(self.samples)
       
        

    def __iter__(self):
        '''
        See example code (ngram) in lecture slides
        '''
        return self

    def __next__(self):
        '''
        See example code (ngram) in slides
        '''
        self.index += 1
        
        if self.index == len (self.samples):
            raise StopIteration
            
        return self.samples[self.index][0],self.samples[self.index][1]

    def shuffle(self):
        '''
        shuffle the data iterator
        '''
        random.shuffle(self.samples)
        
    def standardize (self,rgb_images):
        
        mean = np.mean(rgb_images, axis = (1,2), keepdims= True)
        std = np.std(rgb_images, axis = (1,2), keepdims= True)
        return (rgb_images - mean)/std
    

def main():

    data = Cat_Data()
    model = Cat_Model(activation = sigmoid)  # specify the necessary arguments
    trainer = Cat_Trainer(data, model)
    costs, accuracies = trainer.train(0.01, 500) # experiment with learning rate and number of epochs
    model.save_model()


if __name__ == '__main__':


    
    main()
