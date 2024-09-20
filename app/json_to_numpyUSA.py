import numpy as np
import json 

json_file_path="app/values.json"
def load_values(file_path=json_file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

values = load_values()

# Means and standard deviations of the columns bmi/age
means = values["meansUSA"]
stdevs = values["stdevUSA"]

def transform_yes_noUSA(x):
    if x=='no':
        return -1
    elif x=='yes':
        return 1
    else : 
        return ('Error',x)

def transform_bmiUSA(x):
    if 12<=x<=95:
        return (x-means[0])/stdevs[0]
    else : 
        return ('Error',x)

def transform_phyhealthUSA(x):
    if 0<=x<=30:
        return (x-means[1])/stdevs[1]
    else : 
        return ('Error',x)

def transform_menhealthUSA(x):
    if 0<=x<=30:
        return (x-means[2])/stdevs[2]
    else : 
        return ('Error',x)

def transform_sexUSA(x):
    if x=='Female':
        return -1
    elif x=='Male':
        return 1
    elif x=='Other':
        return 0
    else :
        return ('Error',x)

def transform_ageUSA(x):
    t=0
    if x=='18-24':  
        t=21
    elif x=='25-29':
        t=27
    elif x=='30-34':
        t=32
    elif x=='35-39':
        t=37
    elif x=='40-44':
        t=42
    elif x=='45-49':
        t=47
    elif x=='50-54':
        t=52
    elif x=='55-59':
        t=57
    elif x=='60-64':
        t=62
    elif x=='65-69':
        t=67
    elif x=='70-74':
        t=72
    elif x=='75-79':
        t=77
    elif '80 or older':
        t=80
    else :
        return ('Error',x)
    return (t-means[3])/stdevs[3]  

def transform_diabetesUSA(x):
    if x=='No':
        return -1
    elif x=='Yes':
        return 1
    elif x=='No, borderline diabetes':
        return 0
    else : 
        return ('Error',x)

def transform_genhealthUSA(x):
    t=0
    if x=='Excellent':
        t=4
    elif x=='Very good':
        t=3
    elif x=='Good':
        t=2
    elif x=='Fair':
        t=1
    elif x=='Poor':
        t=0
    else : 
        return ('Error',x)
    return (t-means[4])/stdevs[4] 

def transform_sleeptimeUSA(x):
    if 1<=x<=24:
        return (x-means[5])/stdevs[5] 
    else : 
        return ('Error',x)
    
def json_to_numpyUSA(data):
    transfo_data=np.zeros(16)
    transfo_data[0]= transform_bmiUSA((int(data["Weight"])/((int(data["Height"])/100))**2))
    transfo_data[1]= transform_yes_noUSA(data["Smoking"])
    transfo_data[2]= transform_yes_noUSA(data["AlcoholDrinking"])
    transfo_data[3]= transform_yes_noUSA(data["Stroke"])
    transfo_data[4]= transform_phyhealthUSA(data["PhysicalHealth"])
    transfo_data[5]= transform_menhealthUSA(data["MentalHealth"])
    transfo_data[6]= transform_yes_noUSA(data["DiffWalking"])
    transfo_data[7]= transform_sexUSA(data["Sex"])
    transfo_data[8]= transform_ageUSA(data["Age"])
    transfo_data[9]= transform_diabetesUSA(data["Diabetes"]) 
    transfo_data[10]= transform_yes_noUSA(data["PhysicalActivity"])
    transfo_data[11]= transform_genhealthUSA(data["GenHealth"])
    transfo_data[12]= transform_sleeptimeUSA(int(data["Sleeptime"]))
    transfo_data[13]= transform_yes_noUSA(data["KidneyDisease"])
    transfo_data[14]= transform_yes_noUSA(data["Stroke"])
    transfo_data[15]= transform_yes_noUSA(data["SkinCancer"])
    return transfo_data