# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB

def normalize(data_frame : pd.DataFrame): 
    '''Normalizes the data in a DataFrame, column by column, using min-max aproach\n'''

    normalized_df = pd.DataFrame(data_frame)

    # normalize the data (except the target/classes/results)
    for col in data_frame.columns:
        if col != 'target':
            normalized_df[col] = (data_frame[col] - data_frame[col].min()) / (data_frame[col].max() - data_frame[col].min())

    return normalized_df

def arvores_decisao(data):
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)

    # Operational Phase
    print("Results Using Gini Index:")
      
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    
    conf_matrix = GenerateConfusionMatrix(y_test, y_pred_gini, 2)
    
    print("Nossa matriz: ")
    print(conf_matrix)

    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)

# Function to split the dataset
def splitdataset(data):
  
    # Separating the target variable
    X = data.values[:, :-1]
    Y = data.values[:, -1]
  
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)
      
    return X, Y, X_train, X_test, y_train, y_test

# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
  
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
  
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini
      
# Function to perform training with entropy.
def train_using_entropy(X_train, X_test, y_train):
  
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
  
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

# Function to make predictions
def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
      
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
      
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))

def naive_bayes(data):
    # Separating the target variable
    X = data.values[:, :-1]
    Y = data.values[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    model = GaussianNB()

    model.fit(X_train, y_train) #Fit --> 

    predicted= model.predict(X_test)
    print("Predicted Value: ", predicted)

    cal_accuracy(y_test, predicted)

    '''n_correct = 0
    n_wrong = 0

    for i in range(len(y_test)):
        if y_test[i] == predicted[i]:
            n_correct += 1
        else: 
            n_wrong += 1

    # print the % of correct answers
    print("\nAccuracy: ", (n_correct / (n_correct + n_wrong))*100)'''

    #y_pred = gnb.fit(X_train, y_train).predict(X_test)
    #print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))


def CrossValidation(data_frame: pd.DataFrame, k):
    ''' Get the proportions of the classes\n
        Try to distibute them in k folds in such a way to minimize the diff = |orig prop - fold prop| in all of them\n
        -> put one elem from the 1st class in all folds, then another and another...\n
        -> unitl there aren't any left or len(fold) == n_class_per_fold\n
        -> repeat for the other class unitl there are no elements left'''

    folds = []
    groups = []

    #total_elems = len(data_frame)
    #elems_per_fold = int(np.round(total_elems / k)) 

    grouped_df = data_frame.groupby('DEATH_EVENT', group_keys=False)

    #negative = groups.iloc[0]
    #positive = groups.iloc[1]

    #n_pos = int(np.round((positive/total_elems) * elems_per_fold))
    #n_neg = int(np.round((negative/total_elems) * elems_per_fold))    

    for g in range(len(grouped_df.groups)):
        groups.append(pd.DataFrame()) 
        groups[g] = groups[g].append(grouped_df.get_group(g))
    
    #neg_group = grouped_df.get_group(0)
    #pos_group = grouped_df.get_group(1)
    #print(pos_group)

    for i in range(0, k):
        folds.append(pd.DataFrame())   
    

    for g in range(len(grouped_df.groups)):
        i=0    
        while len(groups[g]) > 0:
            folds[i] = folds[i].append(groups[g].iloc[0], ignore_index=True)
            groups[g] = groups[g].iloc[1:]
            i=(i+1) % k

    # i=0    
    # while len(pos_group) > 0:
    #     folds[i] = folds[i].append(pos_group.iloc[0], ignore_index=True)
    #     pos_group = pos_group.iloc[1:]
    #     i=(i+1) % k
    
    # i=0
    # while len(neg_group) > 0:
    #     folds[i] = folds[i].append(neg_group.iloc[0], ignore_index=True)
    #     neg_group = neg_group.iloc[1:]
    #     i=(i+1) % k   

    # Randomize the element order in each fold / Shuffle folds
    for i in range(0,k):
        folds[i] = folds[i].sample(frac=1)

    #print(folds[2].groupby('DEATH_EVENT', group_keys=False).count())
    return (folds, len(groups))


def GenerateConfusionMatrix(predicted, Y, n_classes):
    
    confusion_matrix = np.zeros(shape=[n_classes, n_classes])

    for i in range(len(predicted)):
        y_pred = int(predicted[i])
        y_verd = int(Y[i])

        confusion_matrix[y_pred, y_verd] += 1

    return confusion_matrix

if __name__ == '__main__':

    data = pd.read_csv("heart_failure_clinical_records_dataset.csv", delimiter=',', header=0)
    data_normalized = normalize(data)
    #data_normalized = data_normalized.iloc[0: , :]
    
    #print("Dataset:", data_normalized.head())

    arvores_decisao(data_normalized)
    #naive_bayes(data_normalized)

    '''
    
    Árvore de decisão
    Naive Bayes
    Regressão Logística
    Florestas Aleatórias
    Boosting


    '''

    folds, n_classes = CrossValidation(data_normalized, 3)
    #print(folds[0])
    
    
    # yt = []
    # for row in test_set.iterrows():
    #     # line by line
    #     # get the data and spit it in to x (atributes) and y (target/classes)
    #     #xt = row[1][:-1]
    #     yt.append(row[1][-1])

    # conf_matrix = GenerateConfusionMatrix(yt, , n_classes)
