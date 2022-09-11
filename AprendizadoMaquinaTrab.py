# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

def normalize(data_frame : pd.DataFrame, train_df : pd.DataFrame, target='target'): 
    '''Normalizes the data in a DataFrame, column by column, using min-max aproach\n'''

    normalized_df = pd.DataFrame(data_frame)

    # normalize the data (except the target/classes/results)
    for col in data_frame.columns:
        if col != target:
            normalized_df[col] = (data_frame[col] - train_df[col].min()) / (train_df[col].max() - train_df[col].min())

    return normalized_df

def arvores_decisao(data, k):
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    #X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    
    folds, _ = generateFolds(data, 'DEATH_EVENT', k)

    accuracy = 0
    precision = 0
    recall = 0 
    f1_measure = 0

    title_folds = []
    array_accuracy_folds = []
    array_precision_folds = []
    array_recall_folds = []
    array_f1_measure_folds = []

    for i in range(k):
        
        y_test = folds[i].iloc[:,-1]
        x_test = folds[i].drop(columns=['DEATH_EVENT'])
        
        x_train = pd.DataFrame()
        for j in range(k):
            if j != i:
                x_train = x_train.append(folds[i])

        y_train = x_train.iloc[:, -1]
        x_train = x_train.drop(columns=['DEATH_EVENT'])

        # Normalize data
        x_test = normalize(x_test, x_train, 'DEATH_EVENT')
        x_train = normalize(x_train, x_train, 'DEATH_EVENT')

        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_test = x_test.to_numpy()
        y_test = y_test.to_numpy()
        
        clf_gini = train_using_gini(x_train, x_test, y_train)
        #clf_entropy = train_using_entropy(x_train, x_test, y_train)

        # Prediction using gini
        y_pred_gini = prediction(x_test, clf_gini)
        cal_accuracy(y_test, y_pred_gini)
        
        conf_matrix = GenerateConfusionMatrix(y_test, y_pred_gini, 2)

        fold_accuracy, fold_precision, fold_recall, fold_f1_measure = generateMetrics(conf_matrix)

        title_folds.append("Fold " + str(i+1)) # Teste
        array_accuracy_folds.append(fold_accuracy) #Teste
        array_precision_folds.append(fold_precision) #Teste
        array_recall_folds.append(fold_recall) #Teste
        array_f1_measure_folds.append(fold_f1_measure) #Teste

        accuracy += fold_accuracy
        precision += fold_precision
        recall += fold_recall
        f1_measure += fold_f1_measure

    accuracy = accuracy/k
    precision = precision/k
    recall = recall/k
    f1_measure = f1_measure/k

    var_acc = calculateVariance(array_accuracy_folds, accuracy)
    var_prec = calculateVariance(array_precision_folds, precision)
    var_rec = calculateVariance(array_recall_folds, recall)
    var_f1 = calculateVariance(array_f1_measure_folds, f1_measure)

    #generateGraphics(title_folds, array_accuracy_folds, array_precision_folds, array_recall_folds, array_f1_measure_folds, "Árvore de decisão")

    testeBoxSplot(array_accuracy_folds, array_precision_folds, array_recall_folds, array_f1_measure_folds)

    print('Nossas estatísticas médias: ')
    print('Accuracy: ', accuracy, end=' | ')
    print('Var acc: ', var_acc)
    print('Precision: ', precision, end=' | ')
    print('Var prec: ', var_prec)
    print('Recall: ', recall, end=' | ')
    print('Var Rec: ', var_rec)
    print('F1_Measure: ', f1_measure, end=' | ')
    print('Var F1: ', var_f1)

    

def generateGraphics(array_title, array_accuracy, array_precision, array_recall, array_f1_measure, metodo):
    f1 = plt.figure(1)
    plt.plot(array_title, array_accuracy, 'k--')
    plt.plot(array_title, array_accuracy, 'go')
    plt.title(metodo + " Accuracy variation")
    f1.show()

    f2 = plt.figure(2)
    plt.plot(array_title, array_precision, 'k--')
    plt.plot(array_title, array_precision, 'go')
    plt.title(metodo + " Precision variation")
    f2.show()

    f3 = plt.figure(3)
    plt.plot(array_title, array_recall, 'k--')
    plt.plot(array_title, array_recall, 'go')
    plt.title(metodo + " Recall variation")
    f3.show()

    f4 = plt.figure(4)
    plt.plot(array_title, array_f1_measure, 'k--')
    plt.plot(array_title, array_f1_measure, 'go')
    plt.title(metodo + " F1 Measure variation")
    f4.show()

def testeBoxSplot(accuracy, precision, recall, f1_measure):
    data = [accuracy, precision, recall, f1_measure]
 
    plt.figure(figsize =(11, 11))
    
    # Creating plot
    bplots = plt.boxplot(data, vert = 1, patch_artist = False, labels=['Accuracy', 'Precision', 'Recall', 'F1_Measure'])

    colors = ['pink', 'lightblue', 'lightgreen', 'red']
    c = 0
    for i, bplot in enumerate(bplots['boxes']):
        bplot.set(color=colors[c], linewidth=3)
        c += 1

    # Adicionando Título ao gráfico
    plt.title("Resultado métricas folds", loc="center", fontsize=18)
    plt.xlabel("Métricas")
    plt.ylabel("Valores")

    # show plot
    plt.show()



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

def naive_bayes(data, k):
    # # Separating the target variable
    # X = data.values[:, :-1]
    # Y = data.values[:, -1]

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    # model = GaussianNB()

    # model.fit(X_train, y_train) #Fit --> 

    # predicted= model.predict(X_test)
    # print("Predicted Value: ", predicted)

    # cal_accuracy(y_test, predicted)

    # #y_pred = gnb.fit(X_train, y_train).predict(X_test)
    # #print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))

    folds, _ = generateFolds(data, 'DEATH_EVENT', k)

    accuracy = 0
    precision = 0
    recall = 0 
    f1_measure = 0

    title_folds = []
    array_accuracy_folds = []
    array_precision_folds = []
    array_recall_folds = []
    array_f1_measure_folds = []

    for i in range(k):
        
        y_test = folds[i].iloc[:,-1]
        x_test = folds[i].drop(columns=['DEATH_EVENT'])
        
        x_train = pd.DataFrame()
        for j in range(k):
            if j != i:
                x_train = x_train.append(folds[i])

        y_train = x_train.iloc[:, -1]
        x_train = x_train.drop(columns=['DEATH_EVENT'])

        # Normalize data
        x_test = normalize(x_test, x_train, 'DEATH_EVENT')
        x_train = normalize(x_train, x_train, 'DEATH_EVENT')

        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_test = x_test.to_numpy()
        y_test = y_test.to_numpy()
        
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        cal_accuracy(y_test, y_pred)
        
        conf_matrix = GenerateConfusionMatrix(y_test, y_pred, 2)

        fold_accuracy, fold_precision, fold_recall, fold_f1_measure = generateMetrics(conf_matrix)

        title_folds.append("Fold " + str(i+1)) # Teste
        array_accuracy_folds.append(fold_accuracy) #Teste
        array_precision_folds.append(fold_precision) #Teste
        array_recall_folds.append(fold_recall) #Teste
        array_f1_measure_folds.append(fold_f1_measure) #Teste

        accuracy += fold_accuracy
        precision += fold_precision
        recall += fold_recall
        f1_measure += fold_f1_measure

    accuracy = accuracy/k
    precision = precision/k
    recall = recall/k
    f1_measure = f1_measure/k

    var_acc = calculateVariance(array_accuracy_folds, accuracy)
    var_prec = calculateVariance(array_precision_folds, precision)
    var_rec = calculateVariance(array_recall_folds, recall)
    var_f1 = calculateVariance(array_f1_measure_folds, f1_measure)

    #generateGraphics(title_folds, array_accuracy_folds, array_precision_folds, array_recall_folds, array_f1_measure_folds, "Árvore de decisão")

    testeBoxSplot(array_accuracy_folds, array_precision_folds, array_recall_folds, array_f1_measure_folds)

    print('Nossas estatísticas médias: ')
    print('Accuracy: ', accuracy, end=' | ')
    print('Var acc: ', var_acc)
    print('Precision: ', precision, end=' | ')
    print('Var prec: ', var_prec)
    print('Recall: ', recall, end=' | ')
    print('Var Rec: ', var_rec)
    print('F1_Measure: ', f1_measure, end=' | ')
    print('Var F1: ', var_f1)

def florestas_aleatorias(data_normalized, k):
    
    folds, _ = generateFolds(data, 'DEATH_EVENT', k)
    model = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth = 3, min_samples_leaf = 5, max_features="sqrt")

    accuracy = 0
    precision = 0
    recall = 0 
    f1_measure = 0

    title_folds = []
    array_accuracy_folds = []
    array_precision_folds = []
    array_recall_folds = []
    array_f1_measure_folds = []

    for i in range(k):
        
        y_test = folds[i].iloc[:,-1]
        x_test = folds[i].drop(columns=['DEATH_EVENT'])
        
        x_train = pd.DataFrame()
        for j in range(k):
            if j != i:
                x_train = x_train.append(folds[i])

        y_train = x_train.iloc[:, -1]
        x_train = x_train.drop(columns=['DEATH_EVENT'])

        # Normalize data
        x_test = normalize(x_test, x_train, 'DEATH_EVENT')
        x_train = normalize(x_train, x_train, 'DEATH_EVENT')

        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_test = x_test.to_numpy()
        y_test = y_test.to_numpy()
        
        model.fit(x_train, y_train)

        # Prediction using gini
        y_pred_gini = model.predict(x_test)
        cal_accuracy(y_test, y_pred_gini)
        
        conf_matrix = GenerateConfusionMatrix(y_test, y_pred_gini, 2)

        fold_accuracy, fold_precision, fold_recall, fold_f1_measure = generateMetrics(conf_matrix)

        title_folds.append("Fold " + str(i+1)) # Teste
        array_accuracy_folds.append(fold_accuracy) #Teste
        array_precision_folds.append(fold_precision) #Teste
        array_recall_folds.append(fold_recall) #Teste
        array_f1_measure_folds.append(fold_f1_measure) #Teste

        accuracy += fold_accuracy
        precision += fold_precision
        recall += fold_recall
        f1_measure += fold_f1_measure

    accuracy = accuracy/k
    precision = precision/k
    recall = recall/k
    f1_measure = f1_measure/k

    var_acc = calculateVariance(array_accuracy_folds, accuracy)
    var_prec = calculateVariance(array_precision_folds, precision)
    var_rec = calculateVariance(array_recall_folds, recall)
    var_f1 = calculateVariance(array_f1_measure_folds, f1_measure)

    #generateGraphics(title_folds, array_accuracy_folds, array_precision_folds, array_recall_folds, array_f1_measure_folds, "Florestas aleatórias")

    print('Nossas estatísticas médias: ')
    print('Accuracy: ', accuracy, end=' | ')
    print('Var acc: ', var_acc)
    print('Precision: ', precision, end=' | ')
    print('Var prec: ', var_prec)
    print('Recall: ', recall, end=' | ')
    print('Var Rec: ', var_rec)
    print('F1_Measure: ', f1_measure, end=' | ')
    print('Var F1: ', var_f1)

def generateFolds(data_frame: pd.DataFrame, target_col='target', k=5):

    import warnings #Remover dps
    warnings.simplefilter(action='ignore', category=FutureWarning) #Remover dps
    # substituir "append" em folds[i] = folds[i].append(...) por "concat" -> python 3.10.0

    ''' Get the proportions of the classes\n
        Try to distibute them in k folds in such a way to minimize the diff = |orig prop - fold prop| in all of them\n
        -> put one elem from the 1st class in all folds, then another and another...\n
        -> unitl there aren't any left or len(fold) == n_class_per_fold\n
        -> repeat for the other class unitl there are no elements left'''

    folds = []
    groups = []

    grouped_df = data_frame.groupby(target_col, group_keys=False)
  

    for g in range(len(grouped_df.groups)):
        groups.append(pd.DataFrame()) 
        groups[g] = groups[g].append(grouped_df.get_group(g))
        

    for i in range(0, k):
        folds.append(pd.DataFrame())   
    

    for g in range(len(grouped_df.groups)):
        i=0    
        while len(groups[g]) > 0:
            folds[i] = folds[i].append(groups[g].iloc[0], ignore_index=True)
            groups[g] = groups[g].iloc[1:]
            i=(i+1) % k


    # Randomize the element order in each fold / Shuffle folds
    for i in range(0,k):
        folds[i] = folds[i].sample(frac=1)

        # # Put the target column in the back as the last column
        # temp_cols = folds[i].columns.tolist()
        # new_cols = temp_cols[1:] + temp_cols[0:1]
        # folds[i] = folds[i][new_cols].reset_index(drop=True)

    #print(folds[2].groupby('DEATH_EVENT', group_keys=False).count())
    return (folds, len(groups))

def calculateVariance(score_lst, mean):
    variance = 0

    for score in score_lst:
        variance += np.square(score - mean)

    return variance / len(score_lst)

def GenerateConfusionMatrix(predicted, Y, n_classes):
    
    confusion_matrix = np.zeros(shape=[n_classes, n_classes])
    
    for i in range(len(predicted)):
        y_pred = int(predicted[i])
        y_verd = int(Y[i])

        confusion_matrix[y_pred, y_verd] += 1

    return confusion_matrix

def generateMetrics(conf_matrix):
    vp = conf_matrix[0][0]
    vn = conf_matrix[1][1]
    fp = conf_matrix[1][0]
    fn = conf_matrix[0][1]


    accuracy = (vp + vn)/ (vp + vn + fp + fn)
    precision = vp / (vp + fp)
    recall = vp / (vp + fn) 
    f1_measure = (2 * precision * recall) / (precision + recall)

    # print('Nossas estatísticas: ')
    # print('Accuracy: ', accuracy)
    # print('Precision: ', precision)
    # print('Recall: ', recall)
    # print('F1_Measure: ', f1_measure)
    # print('\n')

    return (accuracy, precision, recall, f1_measure)

if __name__ == '__main__':

    data = pd.read_csv("heart_failure_clinical_records_dataset.csv", delimiter=',', header=0)
    #data_normalized = normalize(data)
    #data_normalized = data_normalized.iloc[0: , :]
    
    #print("Dataset:", data_normalized.head())

    print("ARVORES DE DECISAO:")
    arvores_decisao(data, 5)
    print("\n\n\n")

    print("FLORESTAS ALEATORIAS:")
    florestas_aleatorias(data, 5)
    print("\n\n\n")
    
    print("NAIVE BAYES:")
    naive_bayes(data, 5)
    print("\n\n\n")

    '''
    
    Árvore de decisão
    Naive Bayes
    Florestas Aleatórias
    Regressão Logística

    '''
    
