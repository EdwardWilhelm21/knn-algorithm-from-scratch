#imports
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, LeaveOneOut
from dis import dis
from statistics import mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix

#LOAD DATA

#Fish File
#fill target array with unique integer corrosponding to fish species
fish_df = pd.read_csv('Fish.csv')
fish_targets = []
for i in range(len(fish_df)):
    if fish_df.iloc[i, 0] == 'Bream':
        fish_targets.append(0)
    elif fish_df.iloc[i, 0] == 'Roach':
        fish_targets.append(1)
    elif fish_df.iloc[i, 0] == 'Whitefish':
        fish_targets.append(2)
    elif fish_df.iloc[i, 0] == 'Parkki':
        fish_targets.append(3)
    elif fish_df.iloc[i, 0] == 'Perch':
        fish_targets.append(4)
    elif fish_df.iloc[i, 0] == 'Pike':
        fish_targets.append(5)
    elif fish_df.iloc[i, 0] == 'Smelt':
        fish_targets.append(6)

#drop column containing string labels for species, add remaining datapoints to list
fish_df = fish_df.drop(columns='Species')
fish_data = fish_df.values.tolist()

#add column to end of dataframe containing int labels for species
fish_df["Species"] = fish_targets

x = fish_data
y = fish_targets

##############################################################

#Iris Data
idata = load_iris()

iris_data = idata['data']
iris_targets = idata['target']

x2 = iris_data
y2 = iris_targets

#####################################################################

#DISTANCE FUNCTIONS

def M_dist(pt, test_pt):
    d = 0
    for f in range(0,len(pt)):
        d += abs(pt[f] - test_pt[f])
    return d
def E_dist(pt, test_pt):
    d = 0
    for f in range(0, len(pt)):
        d += (pt[f] - test_pt[f])**2
    return math.sqrt(d)
  
#####################################################################  
  
#NEIGHBOR FUNCTION

def get_neighbors(data, test_pt, k, metric='euclidean'):

    if metric == "manhattan":
        nD = [(i, M_dist(test_pt, data_pt)) for i, data_pt in enumerate(data)]
    else:
        nD = [(i, E_dist(test_pt, data_pt)) for i, data_pt in enumerate(data)]

    nD.sort(key=lambda x: x[1])
    return [i[0] for i in nD[:k]]

#####################################################################

#KNN ALGORITHM

def run_kNN(data, targets, test_points, k, metric='euclidean', regression=False):

    met = metric
    num = k
    classifications = []
    regressions = []

    if len(data[0]) != len(test_points[0]):
        print("Unequal dimensions")
        return
    
    
    #iterate through each test point
    for j in range(len(test_points)):

        #store k closest neighbors' index in list
        neighbor_index = []
        neighbor_index = get_neighbors(data, test_points[j], num, met)

        #create arr of neighbors target (its classifier)
        neighbor_targets = []
        for i in range(len(neighbor_index)):
            neighbor_targets.append(targets[neighbor_index[i]])
            
        #print(neighbor_targets)
        #take counts of unique elements and return 2 most common elements w/ their counts [(element1, count1), (element2, count2)]
        #if two unique elements have same count mode = -1
        target_counts = Counter(neighbor_targets).most_common(2)
        
        if len(target_counts) < 2:
            mode = target_counts[0][0]
        elif target_counts[0][1] == target_counts[1][1]:
            mode = -1
        else:
            mode = target_counts[0][0]
        
        classifications.append(mode)
        regressions.append(np.mean(neighbor_targets))
        
    #print(test_targets)
    if regression == True:
        return regressions
    else:
        return classifications
    
#####################################################################

#CROSS VALIDATION OF KNN
#fold and test/train split
def evaluate_kNN(data, targets, sklearn = False, numfolds=3, k=1, regression = False, metric='euclidean', matrix=False):
    #permutate data and traget
    my_index = list(range(len(targets)))
    my_index = np.random.permutation(my_index)

    accuracy_sum = 0
    for fold in range(numfolds):
        train_data = []
        train_targets = []
        test_data = []
        test_targets = []
    
        for i in range(len(data)):
            if( i %numfolds == fold):
                test_data.append(data[my_index[i]])
                test_targets.append(targets[my_index[i]])
            else:
                train_data.append(data[my_index[i]])
                train_targets.append(targets[my_index[i]])
                
        #convert to the correct type of array for sklearn
        train_data = np.array(train_data)
        train_targets = np.array(train_targets)
        test_data = np.array(test_data)
        test_targets = np.array(test_targets)

        if sklearn == True:
            num = k
            reg = regression
            met = metric
            knn = KNeighborsClassifier(n_neighbors=num, algorithm='brute', metric=met)
            knn.fit(train_data, train_targets)
            test_pred = knn.predict(test_data)
        else:
            num = k
            reg = regression
            met = metric
            test_pred = run_kNN(train_data, train_targets, test_data, k = num, metric = met, regression = reg)
        
        accuracy = sum([1 for i in range(len(test_pred)) if test_pred[i] == test_targets[i]])/len(test_pred)
        accuracy_sum += accuracy
        
        if matrix==True:
            print(confusion_matrix(test_targets, test_pred))
        
    accuracy_score = accuracy_sum / numfolds
    print(accuracy_score)
    
def run_tests():
    #test arr for loops to try different k's
    test_k = [1,3,7]
    test_k2 = [1, 3, 5, 7, 9, 11, 15, 20, 25, 30, 35, 40, 50, 60]
    print() 
    print('Accuracy Scores for Fish Data')
    print()
    for i in range(len(test_k)):
        
        num = test_k[i]
        
        print('Number of Neighbors:', num)
        print('--------------------')
        print('Own')
        print()
        print('Normal:')
        evaluate_kNN(x, y, sklearn=False, k=num, regression=False, metric ='euclidean')
        print('Regression:')
        evaluate_kNN(x, y, sklearn=False, k=num, regression=True, metric ='euclidean')
        
        print()
        print('Sklearn')
        print()
        print('Normal:')
        evaluate_kNN(x, y, sklearn=True, k=num, regression=False, metric ='euclidean')
        print('Regression:')
        evaluate_kNN(x, y, sklearn=True, k=num, regression=True, metric ='euclidean')
    
        #manhattan tests
        #evaluate_kNN(x, y, sklearn=False, k=num, regression=False, metric ='manhattan')
        #evaluate_kNN(x, y, sklearn=True, k=num, regression=False, metric ='manhatten')
        
        #evaluate_kNN(x, y, sklearn=False, k=num, regression=True, metric ='manhattan')
        #evaluate_kNN(x, y, sklearn=True, k=num, regression=True, metric ='manhattan')
        print()

    print('Confusion Matrix: k=1')     
    evaluate_kNN(x, y, matrix=True)
    print()    
    pplot1 = pd.plotting.scatter_matrix(fish_df, c=y, figsize=(10,10), hist_kwds={'bins': 20}, s=60, alpha=.8)
    
#####################################################################
    
    print('#####################################################################')
    print() 
    print('Accuracy Scores for Iris Data')
    print()
    for i in range(len(test_k)):
        
        num = test_k[i]
        
        print('Number of Neighbors:', num)
        print('--------------------')
        print('Own')
        print()
        print('Normal:')
        evaluate_kNN(x2, y2, sklearn=False, k=num, regression=False, metric ='euclidean')
        print('Regression:')
        evaluate_kNN(x2, y2, sklearn=False, k=num, regression=True, metric ='euclidean')
        
        print()
        print('Sklearn')
        print()
        print('Normal:')
        evaluate_kNN(x2, y2, sklearn=True, k=num, regression=False, metric ='euclidean')
        print('Regression:')
        evaluate_kNN(x2, y2, sklearn=True, k=num, regression=True, metric ='euclidean')

        #manhattan tests
        #evaluate_kNN(x2, y2, sklearn=False, k=num, regression=False, metric ='manhattan')
        #evaluate_kNN(x2, y2, sklearn=True, k=num, regression=False, metric ='manhattan')
        
        #evaluate_kNN(x2, y2, sklearn=False, k=num, regression=True, metric ='manhattan')
        #evaluate_kNN(x2, y2, sklearn=True, k=num, regression=True, metric ='manhattan')
        print()
        
    print('Confusion Matrix: k=1')    
    evaluate_kNN(x2, y2, matrix=True)
        
    irisDF = pd.DataFrame(x2, columns=idata.feature_names)
    pplot2 = pd.plotting.scatter_matrix(irisDF, c=y2, figsize=(10,10), hist_kwds={'bins': 20}, s=60, alpha=.8)

    print('End of Tests')

    
run_tests() 
        
  
    
    
    
    
    
    
    
    
    
    
    
    
    

