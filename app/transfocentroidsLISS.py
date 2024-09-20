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

# Percentage (or mean value normalized) of each columns in the dataset
percLISS=values["percLISS"]
norm_meansLISS=values["norm_meansLISS"]

def transfo_percentage(x):
    return round(max(50*x+50,0.0),2)

def invtransf_bmi(x):
    return round(stdev_bmi*x+mean_bmi,2)

def invtransf_age(x):
    return round(stdev_age*x+mean_age,2)

def invtransf_hist5(x):
    return round(2*(x+1),2)

def invtransf_hist8(x):
    return round((7/2)*(x+1),2)

# Transform the centroid back to the original format of the dataset, and split it between means (float columns) and perc (yes/no columns)
def invtransfo_centroidLISS(L):
    means=np.empty(7, dtype=object) 
    perc=np.empty(12, dtype=object) 
    perc[0]= transfo_percentage(L[0])
    perc[1]= transfo_percentage(L[1])
    means[0]= invtransf_age(L[2])
    means[1]= invtransf_hist5(L[3])
    means[2]= invtransf_hist5(L[4])
    means[3]= invtransf_bmi(L[5])
    means[4]= invtransf_hist5(L[6])
    means[5]= 4-invtransf_hist5(L[7])  #The interface displays ability to walk rather than difficulty walking 
    perc[2]= transfo_percentage(L[8])
    perc[3]= transfo_percentage(L[9])
    perc[4]= transfo_percentage(L[10])
    perc[5]= transfo_percentage(L[11])
    perc[6]= transfo_percentage(L[12])
    perc[7]= transfo_percentage(L[13])
    perc[8]= transfo_percentage(L[14])
    perc[9]= transfo_percentage(L[15])
    perc[10]= transfo_percentage(L[16])
    perc[11]= transfo_percentage(L[17])
    means[6]= invtransf_hist8(L[18])
    return means.tolist(), perc.tolist()


# Algorithm to unnormalized norm_means
def invtransfo_norm_meansLISS(L):
    means=np.empty(7, dtype=object) 
    means[0]=invtransf_age(L[0])
    means[1]=4*L[1]
    means[2]=4*L[2]
    means[3]=invtransf_bmi(L[3])
    means[4]=4*L[4]
    means[5]=4-4*L[5]  #The interface displays ability to walk rather than difficulty walking 
    means[6]=7*L[6]
    return means.tolist()

