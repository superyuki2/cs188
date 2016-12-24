import numpy as np 
import data_classification_utils
from util import raiseNotDefined
import random

class Perceptron(object):
    def __init__(self, categories, numFeatures):
        """categories: list of strings 
           numFeatures: int"""
        self.categories = categories
        self.numFeatures = numFeatures
        
        """YOUR CODE HERE"""
        self.weights = np.zeros((len(categories), numFeatures))


    def classify(self, sample):
        """sample: np.array of shape (1, numFeatures)
           returns: category with maximum score, must be from self.categories"""
        """YOUR CODE HERE"""
        #y is a category
        scores = np.zeros(len(self.weights)) #length of categories
        
        for i in range(len(self.weights)):
          score = np.dot(sample, (self.weights[i]))
          scores[i] = score

        maxScoreInd = np.argmax(scores) #get index of max score
        return self.categories[maxScoreInd] #category w max score


    def train(self, samples, labels):
        """samples: np.array of shape (numFeatures, numSamples) -> numFeatures arrays of length numSamples
           labels: list of numSamples strings, all of which must exist in self.categories 
           performs the weight updating process for perceptrons by iterating over each sample once."""

        """YOUR CODE HERE"""
        #sample is f, labels is bunch of y's
        for i in range(0, len(samples) - 1):
          maxCat = self.classify(samples[i])
          if labels[i] == maxCat:
            continue
          else:
            self.weights[labels[i]] += samples[i]
            self.weights[int(maxCat)] -= samples[i]


