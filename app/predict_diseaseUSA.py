import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import json 

json_file_path="app/values.json"
def load_values(file_path=json_file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

values = load_values()

### Downloading the dataset 
file_path='Models/normalized_USA_dataset.csv'
normalized_dataset = pd.read_csv(file_path)
norm_dataUSA=normalized_dataset.to_numpy()

m1=norm_dataUSA.shape[0]
m2=norm_dataUSA.shape[1]

# Number of person with heart disease in the dataset 
total_disease=values["total_diseaseUSA"]
total_disease_percentUSA=100*total_disease/m1

#####################

###Selection of the same number of patient with and without heart disease 

# # First we select all the patient with heart disease
# inter_data=np.zeros((2*total_disease, m2))
# c_no=0
# no_disease_index=[]

# for n in range (m1):
#     if norm_dataUSA[n][0] == 1 :
#         inter_data[n-c_no] = norm_dataUSA[n]
#     else : 
#         c_no+=1
#         no_disease_index.append(n) # We store the indexes of patient without heart disease

# # We randomly select the same number of patient without heart disease
# random_selection = random.sample(no_disease_index, total_disease)

# for n in range (total_disease):
#     inter_data[total_disease+n] = norm_dataUSA[random_selection[n]]

# # random.shuffle(f_data) doesn't work so we implement it by hand 
# index=[i for i in range(2*total_disease)]
# random.shuffle(index)
# final_data=np.zeros((2*total_disease, m2))

# for k in range (2*total_disease):
#     final_data[k] = inter_data[index[k]]

######################

### Clustering algorithm on the normalized dataset
    
km = KMeans(n_clusters=values["clusters_number"]).fit(norm_dataUSA)

#centroids of the clusters
centroids = km.cluster_centers_

#number of clusters
clusters_number = km.n_clusters

#clustering labels
cluster_labels = km.labels_

### Getting the average of each column in final_data

# Algorithm to transform the result of yes/no or male/female column into a perrcentage

def transfo_percentage(x):
    return round(max(50*x+50,0.0),2)

# Algorithm to compute the size of each cluster

def occurence (x,T):
    c=0
    for k in T :
        if k == x :
            c+=1
    return c

cluster_sizes=np.zeros(clusters_number)

for k in range (clusters_number):
   cluster_sizes[k]=occurence(k,cluster_labels)

# Algorithm to compute the average of each columns

tot_percent_columnUSA=[]
index_column_ynUSA=values["index_column_ynUSA"]

for k in index_column_ynUSA : 
    perc=0
    for i in range (clusters_number):
        perc+=transfo_percentage(centroids[i][k])*cluster_sizes[i]
    perc=perc/(norm_dataUSA.shape[0])
    tot_percent_columnUSA.append(round(perc,2))

tot_percent_columnUSA[6]=round(100-tot_percent_columnUSA[6],2) #do physical activities becomes don't do physical activities

### Deleting heart disease info from all the centroids

centroids_bis=np.zeros((clusters_number, m2-1))

for k in range (clusters_number):
    for j in range (m2-1):
        centroids_bis[k][j]=centroids[k][j+1]

X_test_bis=np.zeros((norm_dataUSA.shape[0], m2-1))

for k in range (norm_dataUSA.shape[0]):
    for j in range (m2-1):
        X_test_bis[k][j]=norm_dataUSA[k][j+1]

###  Algorithm to assign a cluster to a patient's data 

def assign_clusters(v):
    distances = []
    for k in range (clusters_number):
        distance = np.linalg.norm(v-centroids_bis[k])
        distances.append(distance)
        label = np.argmin(distances)
    return label

### Algorithm to determine the cluster with high risk of heart disease 

def is_in (x,T):
    n =len(T)
    for k in range (n):
        if x==T[k]:
            return 1
    return 0

disease_percentUSA=[transfo_percentage(centroids[k][0]) for k in range(clusters_number)]

high_risk_clustersUSA=[]   
low_risk_clustersUSA=[]   

for i in range (clusters_number):
    if disease_percentUSA[i]>=values["thresholdUSA"]*total_disease_percentUSA:
        high_risk_clustersUSA.append(i)
    elif disease_percentUSA[i]<=(1/values["thresholdUSA"])*total_disease_percentUSA:
        low_risk_clustersUSA.append(i)

### Extracting the important component of the clusters

threshold=values["thresholdUSA"]

def clusters_analysisUSA(k):
    L=[]
    index_list_means=[]
    index_list_perc=[0]
    if is_in(k, high_risk_clustersUSA):
        L.append('You are in a cluster with high risk of heart disease.')
    elif is_in(k, low_risk_clustersUSA): 
        L.append('You are in a cluster with low risk of heart disease.')
    else:
        L.append('You are in a cluster with moderate risk of heart disease.')
    L.append('In this cluster:')
    if centroids[k][1]>threshold:
        L.append('- People have an higher BMI than the average')
        index_list_means.append(1)
    elif centroids[k][1]<-threshold:
        L.append('- People have a lower BMI than the average')
        index_list_means.append(1)
    if transfo_percentage(centroids[k][2])>threshold*tot_percent_columnUSA[0]:
        L.append('- People smoke more than the average')
        index_list_perc.append(1)
    elif transfo_percentage(centroids[k][2])<(1/threshold)*tot_percent_columnUSA[0]:
        L.append('- People smoke less than the average')
        index_list_perc.append(1)
    if transfo_percentage(centroids[k][3])>threshold*tot_percent_columnUSA[1]:
        L.append('- People drink more than the average')
        index_list_perc.append(2)
    elif transfo_percentage(centroids[k][3])<(1/threshold)*tot_percent_columnUSA[1]:
        L.append('- People drink less than the average')
        index_list_perc.append(2)
    if transfo_percentage(centroids[k][4])>threshold*tot_percent_columnUSA[2]:
        L.append('- People have had more stroke than the average')
        index_list_perc.append(3)
    elif transfo_percentage(centroids[k][4])<(1/threshold)*tot_percent_columnUSA[2]:
        L.append('- People have had less stroke than the average')
        index_list_perc.append(3)
    if centroids[k][5]>threshold:
        L.append('- People have a poor physical health')
        index_list_means.append(1)
    elif centroids[k][5]<-threshold:
        L.append('- People have a poor physical health')
        index_list_means.append(1)
    if centroids[k][6]>threshold:
        L.append('- People have a poor mental health')
        index_list_means.append(2)
    elif centroids[k][6]<-threshold:
        L.append('- People have a poor mental health')
        index_list_means.append(2)
    if transfo_percentage(centroids[k][7])>threshold*tot_percent_columnUSA[3]:
        L.append('- People have more difficulty to walk or climb stairs than the average')
        index_list_perc.append(4)
    elif transfo_percentage(centroids[k][7])<(1/threshold)*tot_percent_columnUSA[3]:
        L.append('- People have less difficulty to walk or climb stairs than the average')
        index_list_perc.append(4)
    if transfo_percentage(centroids[k][8])>values["sup_threshold_sex"]:
        L.append('- Most people are males')
        index_list_perc.append(5)
    elif transfo_percentage(centroids[k][8])<values["inf_threshold_sex"]:
        L.append('- Most people are females')
        index_list_perc.append(5)
    if centroids[k][9]>threshold:
        L.append('- People are older than the average')
        index_list_means.append(3)
    elif centroids[k][9]<-threshold:
        L.append('- People are younger than the average')
        index_list_means.append(3)
    if transfo_percentage(centroids[k][10])>threshold*tot_percent_columnUSA[5]:
        L.append('- People have more diabetes than the average')
        index_list_perc.append(6)
    elif transfo_percentage(centroids[k][10])<(1/threshold)*tot_percent_columnUSA[5]:
        L.append('-  People have less diabetes than the average')
        index_list_perc.append(6)
    if (100-transfo_percentage(centroids[k][11]))>threshold*(tot_percent_columnUSA[6]):
        L.append('- People do less physical activities than the average')
        index_list_perc.append(7)
    elif (100-transfo_percentage(centroids[k][11]))<(1/threshold)*(tot_percent_columnUSA[6]):
        L.append('- People do more physical activities than the average')
        index_list_perc.append(7)
    if centroids[k][12]>threshold:
        L.append('- People have have a better general health than the average')
        index_list_means.append(4)
    elif centroids[k][12]<-threshold:
        L.append('- People have have a worse general health than the average')
        index_list_means.append(4)
    if centroids[k][13]>threshold:
        L.append('- People sleep more than the average')
        index_list_means.append(5)
    elif centroids[k][13]<-threshold:
        L.append('- People sleep less than the average')
        index_list_means.append(5)
    if transfo_percentage(centroids[k][14])>threshold*tot_percent_columnUSA[7]:
        L.append('- People have more asthma than the average')
        index_list_perc.append(8)
    elif transfo_percentage(centroids[k][14])<(1/threshold)*tot_percent_columnUSA[7]:
        L.append('- People have less asthma than the average')
        index_list_perc.append(8)
    if transfo_percentage(centroids[k][15])>threshold*tot_percent_columnUSA[8]:
        L.append('- People have had more kidney disease than the average')
        index_list_perc.append(9)
    elif transfo_percentage(centroids[k][15])<(1/threshold)*tot_percent_columnUSA[8]:
        L.append('- People have had less kidney disease than the average')
        index_list_perc.append(9)
    if transfo_percentage(centroids[k][16])>threshold*tot_percent_columnUSA[9]:
        L.append('- People have had  more skin cancer than the average')
        index_list_perc.append(10)
    elif transfo_percentage(centroids[k][16])<(1/threshold)*tot_percent_columnUSA[9]:
        L.append('- People have had less skin cancer than the average')
        index_list_perc.append(10)
    return(L, index_list_means, index_list_perc)

def analyze_individual_dataUSA(data):
    label=assign_clusters(data)
    analysis, index_list_means, index_list_perc=clusters_analysisUSA(label)
    return centroids[label], analysis, index_list_means, index_list_perc