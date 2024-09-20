import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json 

json_file_path="app/values.json"
def load_values(file_path=json_file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

values = load_values()

file_path='Models/normalized_LISS_dataset.csv'
normalized_dataset = pd.read_csv(file_path)
norm_dataLISS=normalized_dataset.to_numpy()

# Number of patient with heart disease 
total_disease=values["total_diseaseLISS"]

# Percentage (or mean value normalized) of each columns in the dataset
percLISS=values["percLISS"]
norm_meansLISS=values["norm_meansLISS"]

### Clustering algorithm on the normalized dataset
        
km = KMeans(n_clusters=values["clusters_number"]).fit(norm_dataLISS)

#centroids of the clusters
centroids = km.cluster_centers_

#number of clusters
clusters_number = km.n_clusters

#clustering labels
cluster_labels = km.labels_

# Algorithm to transform the result of yes/no or male/female column into a perrcentage

def transfo_percentage(x):
    return round(max(50*x+50,0.0),2)

disease_percent=[transfo_percentage(centroids[k][0]) for k in range(clusters_number)]

# Function to determine if an element is in a list
def is_in (x,T):
    n =len(T)
    for k in range (n):
        if x==T[k]:
            return 1
    return 0

# Algorithm to find the cluster with/without a majority of patient with heart disease

total_disease_percentLISS= 100*total_disease/norm_dataLISS.shape[0]

high_risk_clustersLISS=[]   
low_risk_clustersLISS=[]   

for i in range (clusters_number):
    if disease_percent[i]>=values["thresholdLISS"]*total_disease_percentLISS:
        high_risk_clustersLISS.append(i)
    elif disease_percent[i]<=(1/values["thresholdLISS"])*total_disease_percentLISS:
        low_risk_clustersLISS.append(i)

### Deleting heart disease info from all the centroids

centroids_bis=np.zeros((clusters_number, 18))

for k in range (clusters_number):
    for j in range (18):
        centroids_bis[k][j]=centroids[k][j+1]

X_test_bis=np.zeros((norm_dataLISS.shape[0], 18))

for k in range (norm_dataLISS.shape[0]):
    for j in range (18):
        X_test_bis[k][j]=norm_dataLISS[k][j+1]

###  Algorithm to assign a cluster to a patient's data 

def assign_clusters(v):
    distances = []
    for k in range (clusters_number):
        distance = np.linalg.norm(v-centroids_bis[k])
        distances.append(distance)
        label = np.argmin(distances)
    return label

### Extracting the important component of the clusters
threshold=values["thresholdLISS"]

def clusters_analysisLISS(k):
    L=[]
    index_list_means=[]
    index_list_perc=[0]
    if is_in(k, high_risk_clustersLISS):
        L.append('You are in a cluster with high risk of heart disease.')
    elif is_in(k, low_risk_clustersLISS): 
        L.append('You are in a cluster with low risk of heart disease.')
    else:
        L.append('You are in a cluster with moderate risk of heart disease.')
    L.append('In this cluster:')
    if transfo_percentage(centroids[k][1])>values["sup_threshold_sex"]:
        L.append('- Most people are males')
        index_list_perc.append(1)
    elif transfo_percentage(centroids[k][1])<values["inf_threshold_sex"]:
        L.append('- Most people are females')
        index_list_perc.append(1)
    if centroids[k][2]>threshold:
        L.append('- People are older than the average')
        index_list_means.append(0)
    elif centroids[k][2]<-threshold:
        L.append('- People are younger than the average')
        index_list_means.append(0)
    if transfo_percentage(centroids[k][3])>100*threshold*norm_meansLISS[1]:
        L.append('- People have have a better general health than the average')
        index_list_means.append(1)
    elif transfo_percentage(centroids[k][3])<(100/threshold)*norm_meansLISS[1]:
        L.append('- People have have a worse general health than the average')
        index_list_means.append(1)
    if transfo_percentage(centroids[k][4])>100*threshold*norm_meansLISS[2]:
        L.append('- People have have a better mental health than the average')
        index_list_means.append(2)
    elif transfo_percentage(centroids[k][4])<(100/threshold)*norm_meansLISS[2]:
        L.append('- People have have a worse mental health than the average')
        index_list_means.append(2)
    if centroids[k][5]>threshold:
        L.append('- People have an higher BMI than the average')
        index_list_means.append(3)
    elif centroids[k][5]<-threshold:
        L.append('- People have a lower BMI than the average')
        index_list_means.append(3)
    if transfo_percentage(centroids[k][6])>100*threshold*norm_meansLISS[4]:
        L.append('- People have have a better physical health than the average')
        index_list_means.append(4)
    elif transfo_percentage(centroids[k][6])<(100/threshold)*norm_meansLISS[4]:
        L.append('- People have have a worse physical health than the average')
        index_list_means.append(4)
    if transfo_percentage(centroids[k][7])>100*threshold*norm_meansLISS[5]:
        L.append('- People have more difficulty to walk or climb stairs than the average')
        index_list_means.append(5)
    elif transfo_percentage(centroids[k][7])<(100/threshold)*norm_meansLISS[5]:
        L.append('- People have less difficulty to walk or climb stairs than the average')
        index_list_means.append(5)
    if transfo_percentage(centroids[k][8])>threshold*percLISS[2]:
        L.append('- People have more sleeping problems than the average')
        index_list_perc.append(2)
    elif transfo_percentage(centroids[k][8])<(1/threshold)*percLISS[2]:
        L.append('- People have less sleeping problems than the average')
        index_list_perc.append(2)
    if transfo_percentage(centroids[k][9])>threshold*percLISS[3]:
        L.append('- People have more high blood pressure problems than the average')
        index_list_perc.append(3)
    elif transfo_percentage(centroids[k][9])<(1/threshold)*percLISS[3]:
        L.append('- People have less high blood pressure problems than the average')
        index_list_perc.append(3)
    if transfo_percentage(centroids[k][10])>threshold*percLISS[4]:
        L.append('- People have more cholesterol than the average')
        index_list_perc.append(4)
    elif transfo_percentage(centroids[k][10])<(1/threshold)*percLISS[4]:
        L.append('- People have less cholesterol than the average')
        index_list_perc.append(4)
    if transfo_percentage(centroids[k][11])>threshold*percLISS[5]:
        L.append('- People have had more stroke than the average')
        index_list_perc.append(5)
    elif transfo_percentage(centroids[k][11])<(1/threshold)*percLISS[5]:
        L.append('- People have had less stroke than the average')
        index_list_perc.append(5)
    if transfo_percentage(centroids[k][12])>threshold*percLISS[6]:
        L.append('- People have more diabetes than the average')
        index_list_perc.append(6)
    elif transfo_percentage(centroids[k][12])<(1/threshold)*percLISS[6]:
        L.append('- People have less diabetes than the average')
        index_list_perc.append(6)
    if transfo_percentage(centroids[k][13])>threshold*percLISS[7]:
        L.append('- People have had more chronic lung disease than the average')
        index_list_perc.append(7)
    elif transfo_percentage(centroids[k][13])<(1/threshold)*percLISS[7]:
        L.append('- People have had less chronic lung disease than the average')
        index_list_perc.append(7)
    if transfo_percentage(centroids[k][14])>threshold*percLISS[8]:
        L.append('- People have more asthma than the average')
        index_list_perc.append(8)
    elif transfo_percentage(centroids[k][14])<(1/threshold)*percLISS[8]:
        L.append('- People have less asthma than the average')
        index_list_perc.append(8)
    if transfo_percentage(centroids[k][15])>threshold*percLISS[9]:
        L.append('- People have more arthritis than the average')
        index_list_perc.append(9)
    elif transfo_percentage(centroids[k][15])<(1/threshold)*percLISS[9]:
        L.append('- People have less arthritis than the average')
        index_list_perc.append(9)
    if transfo_percentage(centroids[k][16])>threshold*percLISS[10]:
        L.append('- People have had more cancer than the average')
        index_list_perc.append(10)
    elif transfo_percentage(centroids[k][16])<(1/threshold)*percLISS[10]:
        L.append('- People have had less cancer than the average')
        index_list_perc.append(10)
    if transfo_percentage(centroids[k][17])>threshold*percLISS[11]:
        L.append('- People smoke more than the average')
        index_list_perc.append(11)
    elif transfo_percentage(centroids[k][17])<(1/threshold)*percLISS[11]:
        L.append('- People smoke less than the average')
        index_list_perc.append(11)
    if transfo_percentage(centroids[k][18])>100*threshold*norm_meansLISS[6]:
        L.append('- People drink more than the average')
        index_list_means.append(6)
    elif transfo_percentage(centroids[k][18])<(100/threshold)*norm_meansLISS[6]:
        L.append('- People drink less than the average')
        index_list_means.append(6)
    return(L, index_list_means, index_list_perc)

def analyze_individual_dataLISS(data):
    label=assign_clusters(data)
    analysis, index_list_means, index_list_perc=clusters_analysisLISS(label)
    return centroids[label], analysis, index_list_means, index_list_perc