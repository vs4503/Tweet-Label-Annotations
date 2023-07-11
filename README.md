project-2-csci-720_project1

 Tweet Label Distributions and K-Means Clustering
=========================================

Name of Participants
--------------------
* Alejandrina Jimenez Guzman (aj7354)
* Vaibhav Santurkar (vs4503)

Overview
--------
In this project, we began by plotting a graph of the label distributions and then plotting a degree histogram of the same. Then we performed K-Means clustering
on the label distributions and plotted a graph of the clusters and the centroids used for those clusters. After we obtained the clusters we grouped the tweet messages 
together, stemmed and lemmatized the individual words in the tweet and found the most frequent words in each cluster and printed those words. 

Dataset description
-----------------
A collection of 1000 tweets was obtained, and based on senitmental analysis of the tweet, we annotated them across 12 labels.

Python Modules Used
--------------------
1) JSON
2) NLTK
3) Pandas
4) Sklearn
5) Matplotlib
6) Networkx
7) Counter
Need to run this in command line in order to generate the confusion matrices with multilabel_confusion_matrix:
pip install git+http://github.com/scikit-learn/scikit-learn.git

Setup and Task Distribution
------

1) Tasks 4,5,6 and 11 -> Run pldl.py to generate the intial graph, the data about the graph and the colored clusters graph
2) Tasks 7,9,10,12 and 13 -> Run task_7.py, task_9_and_10.py, task_12.py, and Task_13_pipeline.py to see the respective tasks' results.
