print ("IMPORTING LIBRARIES...")
import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import requests
from requests.auth import HTTPBasicAuth
from time import sleep


#DOWLOADING FILE FROM DROPBOX FIRST TIME
import urllib.request as urllib2
import os.path
import time
import random
while not os.path.exists('dev.csv') or not os.path.exists('oot0.csv'):
    time.sleep (3*random.random()); #Sleeping less than 3 seconds before going to Dropbox - avoid too many students at once.
    if not os.path.exists('dev.csv'):
        print ("DOWLOADING FILE dev.csv FROM DROPBOX BECAUSE LOCAL FILE DOES NOT EXIST!")
        csvfile = urllib2.urlopen("https://www.dropbox.com/s/yn6hvc0x9sjxbsa/dev.csv?dl=1")
        output = open('dev.csv','wb')
        output.write(csvfile.read())
        output.close()
    if not os.path.exists('oot0.csv'):
        print ("DOWLOADING FILE oot0.csv FROM DROPBOX BECAUSE LOCAL FILE DOES NOT EXIST!")
        csvfile = urllib2.urlopen("https://www.dropbox.com/s/i2l3iexmun0bkp2/oot0.csv?dl=1")
        output = open('oot0.csv','wb')
        output.write(csvfile.read())
        output.close()  
#DOWLOADING FILE FROM DROPBOX FIRST TIME

    
print ("LOADING DATASETS...")
df = pd.read_csv("dev.csv") #DEV-SAMPLE
dfo = pd.read_csv("oot0.csv")#OUT-OF-TIME SAMPLE

print ("IDENTIFYING TYPES...")
in_model = []
list_ib = set()  #input binary
list_icn = set() #input categorical nominal
list_ico = set() #input categorical ordinal
list_if = set()  #input numerical continuos (input float)
list_inputs = set()
output_var = 'ob_target'

for var_name in df.columns:
    if re.search('^i',var_name):
        list_inputs.add(var_name)
        print (var_name,"is input")
    if re.search('^ib_',var_name):
        list_ib.add(var_name)
        print (var_name,"is input binary")
    elif re.search('^icn_',var_name):
        list_icn.add(var_name)
        print (var_name,"is input categorical nominal")
    elif re.search('^ico_',var_name):
        list_ico.add(var_name)
        print (var_name,"is input categorical ordinal")
    elif re.search('^if_',var_name):
        list_if.add(var_name)
        print (var_name,"is input numerical continuos (input float)")
    elif re.search('^ob_',var_name):
        output_var = var_name
    else:
        print ("ERROR: unable to identify the type of:", var_name)


print ("STEP 1: DOING MY TRANSFORMATIONS...")

#first model I tried found out I have missing values in the test set.
#I need to deal with them by replacing with -1.
Xo = dfo.set_index('id')

Xo.isnull().sum()
#Median gave a worst prediction
#Xo = Xo.fillna(Xo.median()) 



print ("STEP 2: SELECTING CHARACTERISTICS TO ENTER INTO THE MODEL...")

print ("STEP 3: DEVELOPING THE MODEL...")
#X = df[in_model]
X = df.drop(columns=[output_var])
y = df[output_var]
Xo.isnull().sum()
Xo = Xo.fillna(-1)
Xo.isnull().sum()

    
#model = grid search random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE

def KS(b,a):  
    """Function that received two parameters; first: a binary variable representing 0=good and 1=bad, 
    and then a second variable with the prediction of the first variable, the second variable can be continuous, 
    integer or binary - continuous is better. Finally, the function returns the KS Statistics of the two lists."""
    try:
        tot_bads=1.0*sum(b)
        tot_goods=1.0*(len(b)-tot_bads)
        elements = zip(*[a,b])
        elements = sorted(elements,key= lambda x: x[0])
        elements_df = pd.DataFrame({'probability': b,'gbi': a})
        pivot_elements_df = pd.pivot_table(elements_df, values='probability', index=['gbi'], aggfunc=[sum,len]).fillna(0)
        max_ks = perc_goods = perc_bads = cum_perc_bads = cum_perc_goods = 0
        for i in range(len(pivot_elements_df)):
            perc_goods =  (pivot_elements_df.iloc[i]['len'] - pivot_elements_df.iloc[i]['sum']) / tot_goods
            perc_bads = pivot_elements_df.iloc[i]['sum']/ tot_bads
            cum_perc_goods += perc_goods
            cum_perc_bads += perc_bads
            A = cum_perc_bads-cum_perc_goods
            if abs(A['probability']) > max_ks:
                max_ks = abs(A['probability'])
    except:
        max_ks = 0
    return max_ks

#Instatiar RandomForest
rf = RandomForestClassifier(class_weight={0:1,1:9}, random_state= 2020)
from sklearn.metrics import roc_auc_score    

#Since gridsearch wasnt helping I am going to try a loop.
for i in [81, 71, 51, 51]:
    for j in [100, 350, 500, 1000, 1100, 1250, 1500, 1750, 2000, 2200]:
        X = df.drop(columns=[output_var]).set_index('id')
        y = df[output_var]
        
        feature_selector = RFE(RandomForestClassifier(class_weight={0:1,1:9}, random_state= 2020, n_estimators= j), n_features_to_select=i)
        feature_selector.fit(X,y)
        train_features_select = feature_selector.transform(X)
        test_features_select = feature_selector.transform(Xo)
        
        rf= RandomForestClassifier(class_weight={0:1,1:9}, random_state= 2020, n_estimators= j)
        
        rf.fit(train_features_select, y)
        y_pred = rf.predict_proba(train_features_select)[:,1]
        yo_pred = rf.predict_proba(test_features_select)[:,1]
        
        
        
        print ("STEP 4: ASSESSING THE MODEL...")
        # CALCULATING GINI PERFORMANCE ON DEVELOPMENT SAMPLE

        gini_score = 2*roc_auc_score(y, y_pred)-1
        print ("GINI DEVELOPMENT=", gini_score)


        KS_score = KS(y,y_pred)
        print ("KS DEVELOPMENT=", KS_score) 


        print ("STEP 5: SUBMITTING THE RESULTS...")

        dfo = dfo.reset_index()
        dfo_tosend = dfo[list(['id'])]
        dfo_tosend['pred'] = yo_pred
        dfo = dfo.set_index("id")
        filename = "attempt_with_recursively_selected_features_"+str(i)+"estimators"+str(j)+".csv"
        dfo_tosend.to_csv(filename, sep=',')

        url = 'http://mfalonso.pythonanywhere.com/api/v1.0/uploadpredictions'

        files = {'file': (filename, open(filename, 'rb'))}
        rsub = requests.post(url, files=files, auth=HTTPBasicAuth('max.franco', 'abinbev2012'))
        resp_str = str(rsub.text)
        print(f"with {i} recursively selected features and {j} estimators the result is {resp_str}")
        best_grade = 0
        grade = float(rsub.text[-5:])
        
        #If statement to save best model
        if grade > best_grade:
            best_grade = grade
            best_model = rf
            best_model_file_name = filename
            best_features = test_features_select
        sleep(10)
        







