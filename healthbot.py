import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)








3333333
training = pd.read_csv('Training.csv')
testing= pd.read_csv('Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y


reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder() #categorical into numerical
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
# print(clf.score(x_train,y_train))
# print ("cross result========")
#Cross-validation is a technique used to assess how well a predictive model will generalize to an independent data set.
#It helps to evaluate the performance of a model by training and testing it on multiple subsets of the available data.
scores = cross_val_score(clf, x_test, y_test, cv=3) #cv = no. of substes to split the data
# print (scores)
print (scores.mean())


# model=SVC()
# model.fit(x_train,y_train)
# print("for svm: ")
# print(model.score(x_test,y_test))

importances = clf.feature_importances_ # to determine the importance of each feature (symptom) in the decision-making process
indices = np.argsort(importances)[::-1] # indices of the features sorted in descending order
features = cols

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def calc_condition(exp,days): #Calculate condition severity based on symptoms experienced and days
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item] #sum of severity scores
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription(): #Retrieve symptom descriptions from a CSV file
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]} # symptom name and corresponding description
            description_list.update(_description)




def getSeverityDict(): #Retrieve severity of symptoms from a CSV file
    global severityDictionary
    with open('Symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict(): #Retrieve precautionary measures from a CSV file
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]} # symptom:precuations
            precautionDictionary.update(_prec)


def getInfo(): #Get user information
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t",end="->")
    name=input("")
    print("Hello, ",name)

def check_pattern(dis_list,inp): #Check input pattern #list of diseases, user_input
    pred_list=[]  #store predicted diseases
    inp=inp.replace(' ','_')
    patt = f"{inp}" #regular expression pattern string to match against diseases in dis_list
    regexp = re.compile(patt) #Compiles the regular expression pattern patt into a regular expression object regexp.
    # This object will be used to search for matches in the list of diseases.
    pred_list=[item for item in dis_list if regexp.search(item)] # to find the disease and list od predicted diseases
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]

def sec_predict(symptoms_exp): #predict disease
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train) #model traning
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)} # to store symptoms and corresponding indices
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1 #Sets the value at the index corresponding to the symptom to 1 in the input vector, indicating the presence of that symptom.

    return rf_clf.predict([input_vector]) #The result is returned as an array containing the predicted disease.


def print_disease(node): #Print disease information
    node = node[0] #node parameter is expected to be a list containing a single element, this line extracts the first element of the list to get the actual node.
    val  = node.nonzero()
    disease = le.inverse_transform(val[0]) #transform the indices (val[0]) back to the original disease labels. This assumes that le is a LabelEncoder object fitted to the disease labels.
    return list(map(lambda x:x.strip(),list(disease)))

# tree_to_code function traverses the decision tree recursively, asking the user for symptoms, predicting the disease, and
# calculating severity based on user input and the decision tree's structure.
# It provides an interactive interface for disease prediction and severity assessment based on symptoms.
def tree_to_code(tree, feature_names): #Convert decision tree to code
    tree_ = tree.tree_ #Extracts the underlying decision tree structure from the tree object.
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ] #Feature Names Handling->uses the actual feature if found otherwise undefined!

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:

        print("\nEnter the symptom you are experiencing  \t\t",end="->")
        disease_input = input("")
        conf,cnf_dis=check_pattern(chk_dis,disease_input) #handle user input and select the relevant symptom.
        if conf==1:
            print("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                print(num,")",it)
            if num!=0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
            # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
            # conf_inp = input("")
            # if(conf_inp=="yes"):
            #     break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days=int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")
    # At each node, it checks whether the feature name matches the input symptom (disease_input). If it matches, it sets val to 1; otherwise, it sets it to 0.
    # It then compares val with the threshold of the current node. Depending on the comparison, it recurses either to the left or right child of the current node.
    # When it reaches a leaf node, it prints the predicted disease, asks the user about symptoms, predicts the disease again, and calculates severity.
    def recurse(node, depth): #traverses the decision tree recursively.
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )
            print("Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                print(syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)

            # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            # print("confidence level is " + str(confidence_level))

    recurse(0, 1)
def mode():
    mo=int(input("ENter 1 for text mode and 0 for speech mode"))
    if mo==0:
        readn(getSeverityDict())
        readn(getDescription())
        readn(getprecautionDict())
        readn(getInfo())
        readn(tree_to_code(clf,cols))
    else:
        getSeverityDict()
        getDescription()
        getprecautionDict()
        getInfo()
        tree_to_code(clf,cols)
mode()
print("----------------------------------------------------------------------------------------")