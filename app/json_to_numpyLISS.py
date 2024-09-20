import numpy as np
import json 

json_file_path="app/values.json"
def load_values(file_path=json_file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

values = load_values()

# Means and standard deviations of the columns bmi/age
mean_age = values["mean_ageLISS"]
stdev_age=values["stdev_ageLISS"]
mean_bmi=values["mean_bmiLISS"]
stdev_bmi=values["stdev_bmiLISS"]

# Function to normalize the data received from the interface
def transform_yes_noLISS(x):
    if x=='no':
        return -1
    elif x=='yes':
        return 1
    else : 
        return ('Error',x)

def transform_bmiLISS(t):
    return (t-mean_bmi)/stdev_bmi

def transfo_sexLISS(t):
    if t=='Female':
       s=-1
    elif t=='Male':
        s=1
    else :
        s=0
    return s

def transform_ageLISS(t):
    return (t-mean_age)/stdev_age

# 1 -> general health = good, -1 -> general health = bad
def transform_genhealthLISS(x):
    t=0
    if x=='Excellent':
        t=4
    elif x=='Very good':
        t=3
    elif x=='Good':
        t=2
    elif x=='Moderate':
        t=1
    elif x=='Poor':
        t=0
    else :
        return ('Error',x)
    return 2*(t/4)-1

# 1 -> mental health = good, -1 -> mental health = bad
def transform_mentalhealth_1_2_4LISS(x):
    t=0
    if x=='Never':
        t=5
    elif x=='Seldom':
        t=4
    elif x=='Sometimes':
        t=3
    elif x=='Often':
        t=2
    elif x=='Mostly':
        t=1
    elif x=='Continuously': 
        t=0
    else :
        return ('Error',x)
    return 2*(t/5)-1

def transform_mentalhealth_3_5LISS(x):
    t=0
    if x=='Never':
        t=0
    elif x=='Seldom':
        t=1
    elif x=='Sometimes':
        t=2
    elif x=='Often':
        t=3
    elif x=='Mostly':
        t=4
    elif x=='Continuously': 
        t=5
    else :
        return ('Error',x)
    return 2*(t/5)-1

# 1 -> phymenhealth = good, -1 -> phymenhealth = bad
def transform_phymentalhealthLISS(x):
    t=0
    if x=='Not at all':
        t=5
    elif x=='Hardly':
        t=4
    elif x=='Sometimes':
        t=3
    elif x=='A bit':
        t=2
    elif x=='Quite a lot':
        t=1
    elif x=='Very much': 
        t=0
    else :
        return ('Error',x)
    return 2*(t/5)-1

# 1 -> difficulty walking, -1 -> no difficulty walking
def transform_diffwalkingLISS(x):
    t=0
    if x=='Without any trouble':
        t=0
    elif x=='With some trouble':
        t=1
    elif x=='With a lot of trouble':
        t=2
    elif x=='Only with an aid or the help of others':
        t=3
    elif x=='Unable to': 
        t=4
    else :
        return ('Error',x)
    return 2*(t/4)-1

# 1 -> physical health = good, -1 -> physical health = bad
def transform_phyhealthLISS(x):
    t=0
    if x=='0 days':
        t=4
    elif x=='1 or 2 days':
        t=3
    elif x=='3 to 5 days':
        t=2
    elif x=='5 to 10 days':
        t=1
    elif x=='More than ten day':
        t=0
    else :
        return ('Error',x)
    return 2*(t/4)-1

# 1 -> drink, -1 -> don't drink
def transform_alcoholdrinkingLISS(x):
    t=0
    if x=='Almost every day':
        t=7
    elif x=='Five or six days per week':
        t=6
    elif x=='Three or four days per week':
        t=5
    elif x=='Once or twice a week':
        t=4
    elif x=='Once or twice a month':
        t=3
    elif x=='Once every two months':
        t=2
    elif x=='Once or twice a year':
        t=1
    elif x=='Not at all over the last 12 months':
        t=0
    else :
        return ('Error',x)
    return 2*(t/7)-1

def json_to_numpyLISS(data):
    transfo_data=np.zeros(18)
    transfo_data[0]= transfo_sexLISS(data["Sex"])
    transfo_data[1]= transform_ageLISS(int(data["Age"]))
    transfo_data[2]= transform_genhealthLISS(data["GenHealth"])
    transfo_data[3]= (transform_mentalhealth_1_2_4LISS(data["MentalHealth1"])+transform_mentalhealth_1_2_4LISS(data["MentalHealth2"])+transform_mentalhealth_1_2_4LISS(data["MentalHealth4"])+transform_mentalhealth_3_5LISS(data["MentalHealth3"])+transform_mentalhealth_3_5LISS(data["MentalHealth5"])+0.5*transform_phymentalhealthLISS(data["PhyMenHealth1"])+0.5*transform_phymentalhealthLISS(data["PhyMenHealth2"])+0.5*transform_phymentalhealthLISS(data["PhyMenHealth3"]))/6.5
    transfo_data[4]= transform_bmiLISS((int(data["Weight"])/((int(data["Height"])/100))**2))
    transfo_data[5]= (0.5*transform_phymentalhealthLISS(data["PhyMenHealth1"])+0.5*transform_phymentalhealthLISS(data["PhyMenHealth2"])+0.5*transform_phymentalhealthLISS(data["PhyMenHealth3"])+transform_phyhealthLISS(data["PhysicalHealth"]))
    transfo_data[6]= (transform_diffwalkingLISS(data["DiffWalking1"])+transform_diffwalkingLISS(data["DiffWalking2"])+transform_diffwalkingLISS(data["DiffWalking3"])+transform_diffwalkingLISS(data["DiffWalking4"]))/4
    transfo_data[7]= transform_yes_noLISS(data["SleepPb"])
    transfo_data[8]= transform_yes_noLISS(data["Highbp"])
    transfo_data[9]= transform_yes_noLISS(data["Highchol"])
    transfo_data[10]= transform_yes_noLISS(data["Stroke"])
    transfo_data[11]= transform_yes_noLISS(data["Diabetes"])
    transfo_data[12]= transform_yes_noLISS(data["Chrld"])
    transfo_data[13]= transform_yes_noLISS(data["Asthma"])
    transfo_data[14]= transform_yes_noLISS(data["Arthritis"])
    transfo_data[15]= transform_yes_noLISS(data["Cancer"])
    transfo_data[16]= transform_yes_noLISS(data["Smoking"])
    transfo_data[17]= transform_alcoholdrinkingLISS(data["AlcoholDrinking"])
    return transfo_data
