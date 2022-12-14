# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn import tree

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

    accuracy = 0
    precision = 0
    recall = 0 
    f1_measure = 0

    title_folds = []
    array_accuracy_folds = []
    array_precision_folds = []
    array_recall_folds = []
    array_f1_measure_folds = []

    vn = 0
    vp = 0
    fp = 0
    fn = 0

    folds, _ = generateFolds(data, 'DEATH_EVENT', k)

    for i in range(k):
        
        y_test = folds[i].iloc[:,-1]
        x_test = folds[i].drop(columns=['DEATH_EVENT'])
        
        x_train = pd.DataFrame()
        for j in range(k):
            if j != i:
                x_train = x_train.append(folds[i])

        y_train = x_train.iloc[:, -1]
        x_train = x_train.drop(columns=['DEATH_EVENT'])

        feature_names = x_train.columns

        # Normalize data
        x_test = normalize(x_test, x_train, 'DEATH_EVENT')
        x_train = normalize(x_train, x_train, 'DEATH_EVENT')

        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_test = x_test.to_numpy()
        y_test = y_test.to_numpy()
        
        clf_gini = train_using_gini(x_train, x_test, y_train, feature_names)

        # Prediction using gini
        y_pred_gini = prediction(x_test, clf_gini)
        #cal_accuracy(y_test, y_pred_gini)
        
        conf_matrix = generateConfusionMatrix(y_test, y_pred_gini, 2)

        vn += conf_matrix[0][0]
        vp += conf_matrix[1][1]
        fp += conf_matrix[1][0]
        fn += conf_matrix[0][1]

        print("Arvore de Decis??o Fold " + str(i + 1) + " Confusion Matrix: ")
        print(conf_matrix)
        print("\n")

        fold_accuracy, fold_precision, fold_recall, fold_f1_measure = generateMetrics(conf_matrix)

        title_folds.append("Fold " + str(i+1))
        array_accuracy_folds.append(fold_accuracy)
        array_precision_folds.append(fold_precision)
        array_recall_folds.append(fold_recall)
        array_f1_measure_folds.append(fold_f1_measure)

        accuracy += fold_accuracy
        precision += fold_precision
        recall += fold_recall
        f1_measure += fold_f1_measure

    accuracy = accuracy/k
    precision = precision/k
    recall = recall/k
    f1_measure = f1_measure/k

    vn = int(vn/k)
    vp = int(vp/k)
    fp = int(fp/k)
    fn = int(fn/k)

    matrixNumpy = np.zeros(shape=[2, 2])
    matrixNumpy[0,0] = vn
    matrixNumpy[1,1] = vp
    matrixNumpy[1,0] = fp
    matrixNumpy[0,1] = fn

    var_acc, desvio_padrao_acc = calculateVarianceAndStdDeviation(array_accuracy_folds, accuracy)
    var_prec, desvio_padrao_prec = calculateVarianceAndStdDeviation(array_precision_folds, precision)
    var_rec, desvio_padrao_rec = calculateVarianceAndStdDeviation(array_recall_folds, recall)
    var_f1, desvio_padrao_f1 = calculateVarianceAndStdDeviation(array_f1_measure_folds, f1_measure)

    generateGraphics(title_folds, array_accuracy_folds, array_precision_folds, array_recall_folds, array_f1_measure_folds, "??rvore de decis??o")
    generateBoxSplot('??rvore de Decis??o', array_accuracy_folds, array_precision_folds, array_recall_folds, array_f1_measure_folds)

    print('Nossas estat??sticas m??dias ??rvores de decis??o: ')
    print('Matriz de confus??o m??dia: \n', matrixNumpy)
    print('\nAccuracy: ', accuracy, end=' | ')
    print('Var acc: ', var_acc)
    print('Desvio padr??o acc: ', desvio_padrao_acc)
    print('Precision: ', precision, end=' | ')
    print('Var prec: ', var_prec)
    print('Desvio padr??o prec: ', desvio_padrao_prec)
    print('Recall: ', recall, end=' | ')
    print('Var Rec: ', var_rec)
    print('Desvio padr??o rec: ', desvio_padrao_rec)
    print('F1_Measure: ', f1_measure, end=' | ')
    print('Var F1: ', var_f1)
    print('Desvio padr??o F1: ', desvio_padrao_f1)

def generateGraphics(array_title, array_accuracy, array_precision, array_recall, array_f1_measure, metodo):
    y_min,y_max = 0.45, 1.05
    f1 = plt.figure()

    plt.plot(array_title, array_accuracy, linestyle='-', label="Accuracy")
    plt.plot(array_title, array_accuracy, 'bo')
    
    plt.plot(array_title, array_precision, linestyle='-', label="Precision")
    plt.plot(array_title, array_precision, 'yo')

    plt.plot(array_title, array_recall, linestyle='-', label="Recall")
    plt.plot(array_title, array_recall, 'go')

    plt.plot(array_title, array_f1_measure, linestyle='-', label="F1_Measure")
    plt.plot(array_title, array_f1_measure, 'ro')
    
    plt.ylim(y_min, y_max)
    plt.yticks(np.arange(0.45, 1.05, step=0.05))
    plt.title('M??tricas ' + metodo)
    f1.legend()
    f1.show()
    
def generateBoxSplot(nome_algoritmo, accuracy, precision, recall, f1_measure):
    data = [accuracy, precision, recall, f1_measure]
 
    plt.figure(figsize =(11, 11))
    
    # Creating plot
    bplots = plt.boxplot(data, vert = 1, patch_artist = False, labels=['Accuracy', 'Precision', 'Recall', 'F1_Measure'])

    colors = ['pink', 'lightblue', 'lightgreen', 'red']
    c = 0
    for i, bplot in enumerate(bplots['boxes']):
        bplot.set(color=colors[c], linewidth=3)
        c += 1

    # Adicionando T??tulo ao gr??fico
    plt.title("Resultado m??tricas folds" + " - " + nome_algoritmo, loc="center", fontsize=18)
    plt.xlabel("M??tricas")
    plt.ylabel("Valores")
    plt.ylim((0.45, 1.05))

    # show plot
    plt.show(block=False)

# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train, feature_names):
  
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=5, min_samples_leaf=5)
  
    # Performing training
    clf_gini.fit(X_train, y_train)

    tree_fig = plt.figure(num=5, figsize = (11,11))
    _ = tree.plot_tree(feature_names= feature_names, class_names=['0', '1'], decision_tree=clf_gini)
    tree_fig.savefig("decision_tree.png")

    return clf_gini
      
# Function to make predictions
def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
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

    accuracy = 0
    precision = 0
    recall = 0 
    f1_measure = 0

    title_folds = []
    array_accuracy_folds = []
    array_precision_folds = []
    array_recall_folds = []
    array_f1_measure_folds = []

    vn = 0
    vp = 0
    fp = 0
    fn = 0
    
    folds, _ = generateFolds(data, 'DEATH_EVENT', k)

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

        conf_matrix = generateConfusionMatrix(y_test, y_pred, 2)
        
        vn += conf_matrix[0][0]
        vp += conf_matrix[1][1]
        fp += conf_matrix[1][0]
        fn += conf_matrix[0][1]

        print("Arvore de Decis??o Fold " + str(i + 1) + " Confusion Matrix: ")
        print(conf_matrix)
        print("\n")

        fold_accuracy, fold_precision, fold_recall, fold_f1_measure = generateMetrics(conf_matrix)

        title_folds.append("Fold " + str(i+1)) 
        array_accuracy_folds.append(fold_accuracy) 
        array_precision_folds.append(fold_precision) 
        array_recall_folds.append(fold_recall) 
        array_f1_measure_folds.append(fold_f1_measure) 

        accuracy += fold_accuracy
        precision += fold_precision
        recall += fold_recall
        f1_measure += fold_f1_measure

    accuracy = accuracy/k
    precision = precision/k
    recall = recall/k
    f1_measure = f1_measure/k

    vn = int(vn/k)
    vp = int(vp/k)
    fp = int(fp/k)
    fn = int(fn/k)

    matrixNumpy = np.zeros(shape=[2, 2])
    matrixNumpy[0,0] = vn
    matrixNumpy[1,1] = vp
    matrixNumpy[1,0] = fp
    matrixNumpy[0,1] = fn

    var_acc, desvio_padrao_acc = calculateVarianceAndStdDeviation(array_accuracy_folds, accuracy)
    var_prec, desvio_padrao_prec = calculateVarianceAndStdDeviation(array_precision_folds, precision)
    var_rec, desvio_padrao_rec = calculateVarianceAndStdDeviation(array_recall_folds, recall)
    var_f1, desvio_padrao_f1 = calculateVarianceAndStdDeviation(array_f1_measure_folds, f1_measure)

    generateGraphics(title_folds, array_accuracy_folds, array_precision_folds, array_recall_folds, array_f1_measure_folds, 'Na??ve Bayes')
    generateBoxSplot('Na??ve Bayes', array_accuracy_folds, array_precision_folds, array_recall_folds, array_f1_measure_folds)

    print('Nossas estat??sticas m??dias Naive Bayes: ')
    print('Matriz de confus??o m??dia: \n', matrixNumpy)
    print('\nAccuracy: ', accuracy, end=' | ')
    print('Var acc: ', var_acc)
    print('Desvio padr??o acc: ', desvio_padrao_acc)
    print('Precision: ', precision, end=' | ')
    print('Var prec: ', var_prec)
    print('Desvio padr??o prec: ', desvio_padrao_prec)
    print('Recall: ', recall, end=' | ')
    print('Var Rec: ', var_rec)
    print('Desvio padr??o rec: ', desvio_padrao_rec)
    print('F1_Measure: ', f1_measure, end=' | ')
    print('Var F1: ', var_f1)
    print('Desvio padr??o F1: ', desvio_padrao_f1)

def florestas_aleatorias(data_normalized, k):

    model = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth = 5, min_samples_leaf = 5, max_features="sqrt", random_state=0)

    accuracy = 0
    precision = 0
    recall = 0 
    f1_measure = 0

    title_folds = []
    array_accuracy_folds = []
    array_precision_folds = []
    array_recall_folds = []
    array_f1_measure_folds = []

    vn = 0
    vp = 0
    fp = 0
    fn = 0

    folds, _ = generateFolds(data, 'DEATH_EVENT', k)

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
        #cal_accuracy(y_test, y_pred_gini)
        
        conf_matrix = generateConfusionMatrix(y_test, y_pred_gini, 2)

        vn += conf_matrix[0][0]
        vp += conf_matrix[1][1]
        fp += conf_matrix[1][0]
        fn += conf_matrix[0][1]

        print("Florestas aleat??rias Fold " + str(i + 1) + " Confusion Matrix: ")
        print(conf_matrix)
        print("\n")

        fold_accuracy, fold_precision, fold_recall, fold_f1_measure = generateMetrics(conf_matrix)

        title_folds.append("Fold " + str(i+1)) 
        array_accuracy_folds.append(fold_accuracy) 
        array_precision_folds.append(fold_precision) 
        array_recall_folds.append(fold_recall) 
        array_f1_measure_folds.append(fold_f1_measure) 

        accuracy += fold_accuracy
        precision += fold_precision
        recall += fold_recall
        f1_measure += fold_f1_measure

    accuracy = accuracy/k
    precision = precision/k
    recall = recall/k
    f1_measure = f1_measure/k

    vn = int(vn/k)
    vp = int(vp/k)
    fp = int(fp/k)
    fn = int(fn/k)

    matrixNumpy = np.zeros(shape=[2, 2])
    matrixNumpy[0,0] = vn
    matrixNumpy[1,1] = vp
    matrixNumpy[1,0] = fp
    matrixNumpy[0,1] = fn

    var_acc, desvio_padrao_acc = calculateVarianceAndStdDeviation(array_accuracy_folds, accuracy)
    var_prec, desvio_padrao_prec = calculateVarianceAndStdDeviation(array_precision_folds, precision)
    var_rec, desvio_padrao_rec = calculateVarianceAndStdDeviation(array_recall_folds, recall)
    var_f1, desvio_padrao_f1 = calculateVarianceAndStdDeviation(array_f1_measure_folds, f1_measure)

    generateGraphics(title_folds, array_accuracy_folds, array_precision_folds, array_recall_folds, array_f1_measure_folds, "Florestas aleat??rias")
    generateBoxSplot('Florestas Aleat??rias', array_accuracy_folds, array_precision_folds, array_recall_folds, array_f1_measure_folds)

    print('Nossas estat??sticas m??dias Florestas aleat??rias: ')
    print('Matriz de confus??o m??dia: \n', matrixNumpy)
    print('\nAccuracy: ', accuracy, end=' | ')
    print('Var acc: ', var_acc)
    print('Desvio padr??o acc: ', desvio_padrao_acc)
    print('Precision: ', precision, end=' | ')
    print('Var prec: ', var_prec)
    print('Desvio padr??o prec: ', desvio_padrao_prec)
    print('Recall: ', recall, end=' | ')
    print('Var Rec: ', var_rec)
    print('Desvio padr??o rec: ', desvio_padrao_rec)
    print('F1_Measure: ', f1_measure, end=' | ')
    print('Var F1: ', var_f1)
    print('Desvio padr??o F1: ', desvio_padrao_f1)

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
        folds[i] = folds[i].sample(frac=1, random_state=0)

    return (folds, len(groups))

def calculateVarianceAndStdDeviation(score_lst, mean):
    variance = 0

    for score in score_lst:
        variance += np.square(score - mean)

    return variance / len(score_lst), np.sqrt(variance / len(score_lst))

def generateConfusionMatrix(predicted, Y, n_classes):
    
    confusion_matrix = np.zeros(shape=[n_classes, n_classes])
    
    for i in range(len(predicted)):
        y_pred = int(predicted[i])
        y_verd = int(Y[i])

        confusion_matrix[y_pred, y_verd] += 1

    return confusion_matrix

def generateMetrics(conf_matrix):

    # predito x verdadeiro
    #   0   1
    # 0 vn  fn
    # 1 fp  vp

    vn = conf_matrix[0][0]
    vp = conf_matrix[1][1]
    fp = conf_matrix[1][0]
    fn = conf_matrix[0][1]

    accuracy = (vp + vn)/ (vp + vn + fp + fn)
    precision = vp / (vp + fp)
    recall = vp / (vp + fn) 
    f1_measure = (2 * precision * recall) / (precision + recall)

    return (accuracy, precision, recall, f1_measure)

if __name__ == '__main__':

    data = pd.read_csv("heart_failure_clinical_records_dataset.csv", delimiter=',', header=0)   
    print("Dataset Length: ", len(data))
    print("Dataset Shape: ", data.shape)

    k = int(input("Insira o numero de Folds desejado:\n"))

    print("ARVORES DE DECISAO:")
    arvores_decisao(data, k)
    print("\n\n\n")

    print("FLORESTAS ALEATORIAS:")
    florestas_aleatorias(data, k)
    print("\n\n\n")
    
    print("NAIVE BAYES:")
    naive_bayes(data, k)
    print("\n\n\n")

    plt.show() 