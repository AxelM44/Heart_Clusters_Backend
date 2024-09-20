import numpy as np
import json 

json_file_path="app/values.json"
def load_values(file_path=json_file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

values = load_values()

meansUSA = values["meansUSA"]
stdevsUSA = values["stdevUSA"]

def transfo_percentage(x):
    return round(max(50*x+50,0.0),2)

def invtransf_r1(x):
    return round(stdevsUSA[0]*x+meansUSA[0],2)

def invtransf_r5(x):
    return round(stdevsUSA[1]*x+meansUSA[1])

def invtransf_r6(x):
    return round(stdevsUSA[2]*x+meansUSA[2])

def invtransf_r8(x):
    if -1 <= x < 0  :
        return 'Female'
    elif 0 <= x < 1 :
        return 'Male'

def invtransf_r9(x):
    return stdevsUSA[3]*x+meansUSA[3]


def transfo_percentage(x):
    return round(max(50*x+50,0.0),2)

def invtransf_r12(x):
    return stdevsUSA[4]*x+meansUSA[4]

def invtransf_r13(x):
    return round(stdevsUSA[5]*x+meansUSA[5],1)

# Transform the centroid back to the original format of the dataset, and split it between means (float columns) and perc (yes/no columns)
def invtransfo_centroidUSA(L):
    means=np.empty(6, dtype=object) 
    perc=np.empty(11, dtype=object) 
    perc[0]=transfo_percentage(L[0])
    means[0]=invtransf_r1(L[1])
    perc[1]=transfo_percentage(L[2])
    perc[2]=transfo_percentage(L[3])
    perc[3]=transfo_percentage(L[4])
    means[1]=invtransf_r5(L[5])
    means[2]=invtransf_r6(L[6])
    perc[4]=transfo_percentage(L[7])
    perc[5]=transfo_percentage(L[8])
    means[3]=invtransf_r9(L[9])
    perc[6]=transfo_percentage(L[10])
    perc[7]=round(100-transfo_percentage(L[11]),2)
    means[4]=invtransf_r12(L[12])
    means[5]=invtransf_r13(L[13])
    perc[8]=transfo_percentage(L[14])
    perc[9]=transfo_percentage(L[15])
    perc[10]=transfo_percentage(L[16])
    return means.tolist(), perc.tolist()

