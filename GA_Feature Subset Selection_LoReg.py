# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:21:47 2020

@author: Aditya Agarwal
"""
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  # doctest: +SKIP
scaler = StandardScaler()
from PIL import Image, ImageTk
import tkinter as tk
import pickle
import os
#import rx
#from rx.scheduler import ThreadPoolScheduler
import time


HEIGHT = 800
WIDTH = 800

def avg(l):
    return (sum(l)/float(len(l)))

def getFitness(individual, X, y):
    """
    Feature subset fitness function
    """
    if(individual.count(0) != len(individual)):
        # get index with value 0
        cols = [index for index in range(len(individual)) if individual[index] == 0]
        X_parsed = X.drop(X.columns[cols], axis=1)
        X_subset = pd.get_dummies(X_parsed)
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.30)
        X_test,X_validation,y_test,y_validation = train_test_split(X_test,y_test,test_size=0.33)
        # apply classification algorithm
        clf = LogisticRegression(solver='lbfgs',max_iter = 10000,multi_class='multinomial').fit(X_train,y_train)
        mean_fitness = clf.score(X_test,y_test)
        #nn_model =  tf.keras.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(128, activation='relu'),
                                              #tf.keras.layers.Dense(7)])
        #nn_model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          #from_logits=True),
                      #metrics=['accuracy'])

        #nn_model.fit(X_train.values, y_train, epochs=50)
        #test_loss, test_acc = nn_model.evaluate(X_test,  y_test, verbose=2)
        return mean_fitness
    else:
        return(0,)


def populate(features, size=50):
    initial = []
    for _ in range(size):
        entity = []
        for feature in features:
            val = np.random.randint(0, 2)
            entity.append(val)
        initial.append(entity)

    return np.array(initial)


def mutate(population, mutation_rate):
    mutated_pop = []
    for p in population:
        p_list = p.tolist()
        und = np.random.choice(2, 1, p = [1 - mutation_rate, mutation_rate])
        if(und > 0):
            m_index = np.random.randint(0, len(p_list))
            if p_list[m_index] == 0:
                p_list[m_index] = 1
            else:
                p_list[m_index] = 0
            mutated_pop.append(p_list)
        else:
            mutated_pop.append(p_list)

    return np.array(mutated_pop)


def cross(population, crossover_rate):
    new_pop = population.tolist()
    for _ in range(int(crossover_rate*len(population))):
        p = population[np.random.randint(0, len(population))].tolist()
        m = population[np.random.randint(0, len(population))].tolist()
        entity = p[0:len(p)//2]
        for i in m[len(m)//2:len(m)]:
        	entity.append(i)

        new_pop.append(entity)

    return np.array(new_pop)


def geneticAlgorithm(X, y, n_population, n_generation, mutation_rate, crossover_rate):
    global header
    global maxAccuracy
    maxAccuracy = 0
    global bestindiv
    population = populate(X.columns, n_population)
    a, prev_fitness, b = bestIndividual(population, X, y)
    fitness_history = []
    for i in range(n_generation):
        population = mutate(population, mutation_rate)
        population = cross(population, crossover_rate)
        indiv, current_fitness, head = bestIndividual(population, X, y)
        print(str(i) + "th generation maxfit is : " + str(current_fitness))
        if(current_fitness>maxAccuracy):
            maxAccuracy = current_fitness
            header = head
            bestindiv = indiv
        if(i < int(n_generation/2)):
            continue
        # break if not more than 1% change in fitness values
        if(current_fitness - prev_fitness < 0.01*prev_fitness):
            break
        prev_fitness = current_fitness
        fitness_history.append(current_fitness)
    return bestindiv,fitness_history,header,maxAccuracy

def bestIndividual(hof, X, y):
    """
    Get the best individual
    """
    _individual = []
    maxAccuracy = 0.0
    for individual in hof:
    	individual = individual.tolist()
    	val = getFitness(individual, X, y)
    	if(val > maxAccuracy):
    		maxAccuracy = val
    		_individual = individual

    _individualHeader = [list(X)[i] for i in range(
        len(_individual)) if _individual[i] == 1]
    # _individual = _individual.tolist()
    return _individual, maxAccuracy ,_individualHeader

def getDir(direc):
    global dataFramePath
    dataFramePath = direc
    main()
    
def main():
    ### Some preprocessing ###
    fin_str = ""
    label['text'] = "Please wait while we run the GA."
    df = pd.read_csv(f"{dataFramePath}")
    print(df.head)
    le = LabelEncoder() ## encode emotions
    le.fit(df.iloc[:, -1])
    y = le.transform(df.iloc[:, -1])
    X = df.iloc[:, :-1]
    print(y)
    print(X)
    ### Test with all features ###
    individual = [1 for i in range(len(X.columns))]
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.30)
    clf = LogisticRegression(solver='lbfgs',max_iter = 10000,multi_class='multinomial').fit(X_train,y_train)
    mean_fitness = clf.score(X_test,y_test)
    ### Pickle dump if required ###
    #filename = 'finalized_model_all_features.sav'
    #pickle.dump(clf, open(filename, 'wb'))
    ### Pickled ###
    print("mean accuracy all features : " + str(mean_fitness))
    print("Accuracy with all features: \t" + str(getFitness(individual, X, y)) + "\n")
    fin_str += "\n\nAccuracy with all features: \t" + str(getFitness(individual, X, y)) + "\n"
    label['text'] = fin_str
    ### Parameter variations for the Genetic algorithm ###
    n_pop = 5
    n_gen = 1
    x1 = []
    murate_variation_acc=[]
    mutation_rate = 0.03
    crossover_rate = 0.5
    for murate in range(1,8):
        mutation_rate = murate*0.1
        x1.append(mutation_rate)
        indi,f_hist,header,maxacc = geneticAlgorithm(X, y, n_pop, n_gen, mutation_rate, crossover_rate)
        murate_variation_acc.append(maxacc)
    x2 = []
    print(murate_variation_acc)
    corate_variation_acc=[]
    mutation_rate = 0.05
    crossover_rate = 0.5
    for corate in range(1,8):
        crossover_rate = corate*0.1
        x2.append(crossover_rate)
        indi,f_hist,header,maxacc = geneticAlgorithm(X, y, n_pop, n_gen, mutation_rate, crossover_rate)
        corate_variation_acc.append(maxacc)
    ### PLOTTING ###
    plt.plot(range(n_gen),f_hist, color = 'tab:blue')
    plt.xlim([0,10])
    plt.ylim([0,1])
    plt.xlabel("No of Gen")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Generations")
    plt.show()
    plt.plot(x1,murate_variation_acc, color = 'tab:orange')
    plt.xlim([0,0.1])
    plt.ylim([0,1])
    plt.xlabel("Mutation Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Mutation Rate")
    plt.show()
    plt.plot(x2,corate_variation_acc, color = 'tab:red')
    plt.xlim([0,0.1])
    plt.ylim([0,1])
    plt.xlabel("Crossover Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Crossover Rate")
    plt.show()
    #### END PLOT ####
    print(corate_variation_acc)    
    indi,f_hist,header,maxacc = geneticAlgorithm(X, y, n_pop, n_gen, mutation_rate, crossover_rate)
    print(indi)
    print(f_hist)
    print(header)
    label['text'] = fin_str
    ### Let's train with the best feature subset ###
    X = df[header]
    print(X.head())
    X.to_csv('headers.csv',index=False,header=True)
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.30)
    
    clf = LogisticRegression(solver='lbfgs',max_iter = 10000,multi_class='multinomial').fit(X_train,y_train)
    mean_fitness = clf.score(X_test,y_test)
    ### Pickle dump if required ###
    #filename = 'finalized_model.sav'
    #pickle.dump(clf, open(filename, 'wb'))
    ### Pickle dump end ###
    fin_str+= "\n\nFinaly accuracy on test with feature subset is : " + str(mean_fitness)
    print("Finaly accuracy on test is : ",mean_fitness)
    label['text'] = fin_str

def getQuery(modelname):
    global dataFramePath
    fin_str = ""
    label['text'] = "Please wait while we run the GA.\n"
    #df = pd.read_csv(f"{dataFramePath}")
    df = pd.read_csv('fin.csv')
    print(df.head())
    le = LabelEncoder()
    le.fit(df.iloc[:, -1])
    #y = df.iloc[:,-1]
    y = le.transform(df.iloc[:, -1])
    X = df.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.30)
    print(X)
    print(y_test)
    filename = f'{modelname}_all_features.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X, y)
    print(result)
    fin_str += "\nMean accuracy with all features is : " + str(result)
    X_subset = pd.read_csv("headers.csv")
    print(X_subset.head())
    cols = X_subset.columns
    X_subset = X[cols]
    print(X_subset.head())
    filename = f'{modelname}.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_subset, y)
    print(result)
    fin_str += "\n\nMean accuracy with feature subset is : " + str(result)
    label['text'] = fin_str
    
    
def openReport():
    os.startfile('A2.pdf')

    
root = tk.Tk()

canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

imagex = Image.open('apples.jpg')
photo = ImageTk.PhotoImage(imagex,master=root)
background_label = tk.Label(root, image=photo)
background_label.image = photo
background_label.place(relwidth=1, relheight=1)

frame = tk.Frame(root, bg='#C0C0C0', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.90, relheight=0.065, anchor='n')

entry = tk.Entry(frame, font=40)
entry.place(relwidth=0.65, relheight=0.70)

button = tk.Button(frame, text="Enter CSV Path", font=30, command=lambda: getDir(entry.get()))
button.place(relx=0.7, relheight=0.70, relwidth=0.3)

frame2 = tk.Frame(root, bg='#C0C0C0', bd=5)
frame2.place(relx=0.5, rely=0.22, relwidth=0.90, relheight=0.065, anchor='n') 

entry2 = tk.Entry(frame2, font=40)
entry2.place(relwidth=0.65, relheight=0.70)

button2 = tk.Button(frame2, text="Display Results", font=30, command=lambda: getQuery(entry2.get()))
button2.place(relx=0.7, relheight=0.70, relwidth=0.3)

frame3 = tk.Frame(root, bg='#C0C0C0', bd=5)
frame3.place(relx=0.5, rely=0.34, relwidth=0.45, relheight=0.065, anchor='n') 

button3 = tk.Button(frame3, text="Open results report", font=30, command=lambda: openReport())
button3.place(relx=0.15, relheight=0.90, relwidth=0.7)

lower_frame = tk.Frame(root, bg='#C0C0C0', bd=10)
lower_frame.place(relx=0.5, rely=0.5, relwidth=0.9, relheight=0.4, anchor='n')

label = tk.Label(lower_frame)
label.place(relwidth=1, relheight=1)

root.mainloop()