# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:21:47 2020

@author: Aditya Agarwal
"""
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  # doctest: +SKIP
scaler = StandardScaler()
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf

import sys

def avg(l):
    """
    Returns the average between list elements
    """
    return (sum(l)/float(len(l)))


def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """
    # individual = individual.tolist()
    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(
            len(individual)) if individual[index] == 0]
        # get features subset
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.30)
        # apply classification algorithm
        # clf = LogisticRegression(max_iter = 10000)
        # clf =  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        nn_model=model = tf.keras.Sequential([tf.keras.layers.Flatten(), 
                                              tf.keras.layers.Dense(128, activation='relu'), 
                                              tf.keras.layers.Dense(7)])
        nn_model.compile(optimizer='adam', 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                      metrics=['accuracy'])
        
        nn_model.fit(X_subset, y, epochs=300)
        test_loss, test_acc = nn_model.evaluate(X_subset,  y, verbose=2)
        return test_acc
    else:
        return(0,)

def populate(features, size = 50): 
    initial = []  
    for _ in range(size): 
        entity = [] 
        for feature in features: 
            val = np.random.randint(0,2) 
            entity.append(val)
        initial.append(entity)
        #print(entity) 
      
    return np.array(initial) 

def mutate(population): 
    n = np.random.randint(0, len(population))
    p = population[np.random.randint(0, len(population))]
    l = np.random.randint(0, len(p))
    population[n][l] = np.random.randint(0,2)
    print("\n\ninside mutate" + str(population))
    return population  
  
def cross(population, size = 50): 
    new_pop = [] 
      
    for _ in range(size): 
        und= np.random.normal(0, 1)
        p = population[np.random.randint(0, len(population))].tolist()
        m = population[np.random.randint(0, len(population))].tolist()
        entity = p[0:len(p)//2]
        for i in m[len(m)//2:len(m)]:
        	entity.append(i)
        
        if und>0:
            m_index=np.random.randint(0, len(entity))
            if entity[m_index]==0:
                entity[m_index]=1
            else:
                entity[m_index]=0

        new_pop.append(entity) 
      
    return np.array(new_pop)  

def geneticAlgorithm(X, y, n_population, n_generation):

	#print("X columns" + str(X.columns))

	population = populate(X.columns, n_population)
	
	for _ in range(n_generation): 
		population = cross(population, 50)
	return population

def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
    maxAccurcy = 0.0
    for individual in hof:
    	individual = individual.tolist()
    	val = getFitness(individual, X, y)
    	if(val > maxAccurcy):
    		maxAccurcy = val
    		_individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    #_individual = _individual.tolist()
    return _individual, maxAccurcy ,_individualHeader

# dataFramePath = input("Please enter csv path\n")
n_pop = 10
n_gen = 30
dataFramePath="/home/rudraj1t/eContent/AI/Coding Assignment/labelled-combined.csv"
df = pd.read_csv(f"{dataFramePath}")
# df=pd.read_csv("labelled-combined.csv")
# print(df.head)
le = LabelEncoder()
le.fit(df.iloc[:, -1])
y = le.transform(df.iloc[:, -1])
X = df.iloc[:, :-1]
#print(y)
# print("\nX: ")
# print(X)
# print("\n")

individual = [1 for i in range(len(X.columns))]
print("Accuracy with all features: \t" + str(getFitness(individual, X, y)) + "\n")

hof = geneticAlgorithm(X, y, n_pop, n_gen)

# select the best individual
individual, accuracy, header = bestIndividual(hof, X, y)
#print(individual)
#individual = individual.tolist()
print('Best Accuracy: \t' + str(accuracy))
print('Number of Features in Subset: \t' + str(individual.count(1)))
# print('Individual: \t\t' + str(individual))

X = df[header]

# clf = LogisticRegression(max_iter = 10000)
# clf =  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# scores = cross_val_score(clf, X, y, cv=5)
# print("Accuracy with Feature Subset: \t" + str(avg(scores)) + "\n")

