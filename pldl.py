from collections import Counter
import networkx as nx
from optparse import OptionParser
import pandas as pd
from scipy.stats import multinomial
import matplotlib.pyplot as plt
import numpy as np
import nltk
import string
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import htest

optparser = OptionParser()
optparser.add_option('-f', '--inputFile',
                     dest='input_file',
                     help='json input filename',
                     default="jobQ3_BOTH_train.json")
optparser.add_option('-s', '--sample',
                     dest='sample_size',
                     help='Estimated sample size of each input',
                     default=10,
                     type='float')
optparser.add_option('-c', '--confidence',
                     dest='confidence',
                     help='Confidence (float) of regions desired',
                     default=0.9,
                     type='float')

(options, args) = optparser.parse_args()

df = pd.read_json(options.input_file, orient='split')
Y_dict = (df.groupby('message_id')
    .apply(lambda x: dict(zip(x['worker_id'],x['label_vector'])))
    .to_dict())
Ys = {x: list(y.values()) for x,y in Y_dict.items()}
Yz = {x: Counter(y) for x,y in Ys.items()}
dims = max([max(y.values()) for x,y in Yz.items()])+1
Y = {x:[Yz[x][i] if i in Yz[x] else 0 for i in range(dims)] for x,y in Yz.items()}
labels = df.groupby(['label', 'label_vector']).first().index.tolist()
Yframe = pd.DataFrame.from_dict(Y, orient='index')
XnY = df.groupby("message_id").first().join(Yframe, on="message_id")[['message',0,1,2,3,4,5,6,7,8,9,10,11]]


t = {}
for x,y in Y.items():
    y1 = multinomial(options.sample_size, [yi/sum(y) for yi in y])
    y2 = htest.most_likely(y1)
    t[tuple(y2)] = x

friendlist = []


for x,y in Y.items():
    print (f"x: {x}, y: {y}")
    #my = multinomial(sum(y), [yi/sum(y) for yi in y])
    my = multinomial(options.sample_size, [yi/sum(y) for yi in y])
    mcr = htest.min_conf_reg(my, options.confidence)
    #ldls = [[int(i) for i in m.p * m.n] for m in mcr]
    ldls = [tuple(i) for i in mcr]
    friends = []
    """
    for mc in mcr:
        if tuple(mc) in t:
            friends.append(mc)
    """
    friends = set(ldls) & set(t.keys()) 
    for friend in friends:
        if (tuple(y) != friend):
            friendlist.append((tuple(y),friend))

g = nx.Graph(friendlist)
nx.write_gexf(g,f"label_space_{options.confidence}_{options.sample_size}.gexf")
#Tasks 4, 5 and 6
nx.draw(g)
nodes = g.number_of_nodes();
edges = g.number_of_edges();
connectedcomponents = nx.number_connected_components(g)
print("The number of nodes is: ")
print(nodes)
print("The number of edges is: ")
print(edges)
print("The number of connected components is: ")
print(connectedcomponents)
print("The density of the graph: ")
print(nx.density(g))

degree_sequence = sorted([d for n,d in g.degree()], reverse = True)
degreeCount = Counter(degree_sequence)
deg,cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d+0.4 for d in deg])
ax.set_xticklabels(deg)

plt.show()

#Perform PCA Dimensionality reduction from 12 dimension to 2 dimensions
label_distributions = XnY.drop(["message"], axis=1)
labels = label_distributions.values
pca = PCA(n_components=2)
reduced_Labels = pca.fit_transform(labels)
final_Labels = pd.DataFrame(data = reduced_Labels, columns = ['Label1', 'Label2'])

#Prepare the dataset and set the number of clusters
kmeans_Labels = final_Labels[["Label1", "Label2"]]
no_of_clusters = 12

# Select 12 random data points as centroids 
centroids = (kmeans_Labels.sample(n = no_of_clusters))

#Differnce between new centroid and old centroid
difference = 1

second_index = 0

while(difference != 0):
    
    kmeans_Distance = kmeans_Labels
    
    first_index = 1
    
    #Find distance between data points and centroids
    for index,centroid_rows in centroids.iterrows():
        
        point_Distances=[]
        
        #Calculate the euclidean distance between centroid and data point in question
        for distance_index, distance_rows in kmeans_Distance.iterrows():
            
            firstlabel_distance = (centroid_rows["Label1"] - distance_rows["Label1"])**2
            
            secondlabel_distance = (centroid_rows["Label2"] - distance_rows["Label2"])**2
            
            overall_Distance = np.sqrt(firstlabel_distance + secondlabel_distance) 
            
            point_Distances.append(overall_Distance)
        
        kmeans_Labels[first_index] = point_Distances
        
        first_index = first_index + 1
 
    update_Centroids = []
    
    #Update the centroids and determine which points belong to which cluster by minimum distance
    for index, label_rows in kmeans_Labels.iterrows():
        
        minimum_Distance = label_rows[1]
        
        position = 1
        
        #Check minimum distance of data point 
        for first_index in range(no_of_clusters):
            
            if label_rows[first_index + 1] < minimum_Distance:
                
                minimum_Distance = label_rows[first_index + 1]
                
                position = first_index + 1
        
        update_Centroids.append(position)
    
    kmeans_Labels["Cluster"] = update_Centroids
    
    #Update new centroids to dataframe
    centroids_new = kmeans_Labels.groupby(["Cluster"]).mean()[["Label2","Label1"]]
    
    if second_index == 0:
        
        difference = 1
        
        second_index = second_index + 1
    
    else:
        #Find the new centroids based of the old centroids
        difference = (centroids_new['Label2'] - centroids['Label2']).sum() + (centroids_new['Label1'] - centroids['Label1']).sum()
    
    #Update centroids to dataframe
    centroids = kmeans_Labels.groupby(["Cluster"]).mean()[["Label2","Label1"]]

color=['yellow','#17F91D','purple', 'blue', 'green', '#00E8FD', 'cyan', 'orange', '#7F8A8B', '#DB13EE', '#26E1DA', '#D1D150']

#Plot the clusters and the data points in the scatter plot
for third_index in range(no_of_clusters):
    
    final_Labels = kmeans_Labels[kmeans_Labels["Cluster"] == third_index + 1]
    
    plt.scatter(final_Labels["Label1"],final_Labels["Label2"],c = color[third_index])

plt.scatter(centroids["Label1"],centroids["Label2"],c='red')
plt.xlabel('Label1')
plt.ylabel('Label2')
plt.show()

