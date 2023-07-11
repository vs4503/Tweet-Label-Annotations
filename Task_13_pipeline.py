import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
import re, string, unicodedata
from collections import Counter
from scipy.stats import multinomial
from optparse import OptionParser
import htest
import networkx as nx
import matplotlib.pyplot as plt

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


df = pd.read_csv("enhanced_data.csv")
Y_dict = (df.groupby('message_id')
          .apply(lambda x: dict(zip(x['worker_id'], x['label_vector'])))
          .to_dict())
Ys = {x: list(y.values()) for x, y in Y_dict.items()}
Yz = {x: Counter(y) for x, y in Ys.items()}
dims = max([max(y.values()) for x, y in Yz.items()]) + 1
Y = {x: [Yz[x][i] if i in Yz[x] else 0 for i in range(dims)] for x, y in Yz.items()}
labels = df.groupby(['label', 'label_vector']).first().index.tolist()
Yframe = pd.DataFrame.from_dict(Y, orient='index')
XnY = df.groupby("message_id").first().join(Yframe, on="message_id")[['message', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

t = {}
for x, y in Y.items():
    y1 = multinomial(options.sample_size, [yi / sum(y) for yi in y])
    y2 = htest.most_likely(y1)
    t[tuple(y2)] = x

friendlist = []

for x, y in Y.items():
    print(f"x: {x}, y: {y}")
    # my = multinomial(sum(y), [yi/sum(y) for yi in y])
    my = multinomial(options.sample_size, [yi / sum(y) for yi in y])
    mcr = htest.min_conf_reg(my, options.confidence)
    # ldls = [[int(i) for i in m.p * m.n] for m in mcr]
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
            friendlist.append((tuple(y), friend))

g = nx.Graph(friendlist)
nx.write_gexf(g, f"label_space_{options.confidence}_{options.sample_size}.gexf")
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

degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
degreeCount = Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)

plt.show()



# Task 9

df = pd.read_csv("enhanced_data.csv")
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


X = np.array(list(Y.values()))


km = KMeans(
    n_clusters=12, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)

XnY['cluster'] = y_km
# generate .cvs file where each message is paired with it corresponding cluster

compression_opts = dict(method='zip',
                        archive_name='clustered_data.csv')
XnY.to_csv('clustered_enhanced_data.zip', index=False, compression=compression_opts)
########################################################################################################################
# taks 10


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(10)

mess_lst = XnY.drop([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], axis=1).values.tolist()

cluster_1 = []
cluster_2 = []
cluster_3 = []
cluster_4 = []
cluster_5 = []
cluster_6 = []
cluster_7 = []
cluster_8 = []
cluster_9 = []
cluster_10 = []
cluster_11 = []
cluster_12= []
for x in range(len(mess_lst)):
  if mess_lst[x][1]==0:
    cluster_1.append(mess_lst[x][0])
  elif mess_lst[x][1]==1:
    cluster_2.append(mess_lst[x][0])
  elif mess_lst[x][1]==2:
    cluster_3.append(mess_lst[x][0])
  elif mess_lst[x][1]==3:
    cluster_4.append(mess_lst[x][0])
  elif mess_lst[x][1]==4:
    cluster_5.append(mess_lst[x][0])
  elif mess_lst[x][1]==5:
    cluster_6.append(mess_lst[x][0])
  elif mess_lst[x][1]==6:
    cluster_7.append(mess_lst[x][0])
  elif mess_lst[x][1]==7:
    cluster_8.append(mess_lst[x][0])
  elif mess_lst[x][1]==8:
    cluster_9.append(mess_lst[x][0])
  elif mess_lst[x][1]==9:
    cluster_10.append(mess_lst[x][0])
  if mess_lst[x][1]==10:
    cluster_11.append(mess_lst[x][0])
  else:
    cluster_12.append(mess_lst[x][0])

# Preprocessing
# Downloading the tools necessary to process the data
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

# The following functions are from the Medium Post:
# https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258
# function to convert nltk tag to wordnet tag


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)


stop = set(stopwords.words('english'))

# after analysing the content of some of the post we realized there are some words that are often used to ask questions
# or to request help with some problem, therefore we added those words as extra stopwords

extra_stopwrods = ['make', 'have', 'well', 'way', 'x200b', 'also', 'know', 'try', 'think', 'could', 'would', 'like',
                   'want', 'use', 'get', 'need','gt','na','lol','someone']

for i in extra_stopwrods:
    stop.add(i)

def get_meta_tweet(lst):
    clean_text = []
    process_str = remove_URL(lst)
    token_str = tokenizer.tokenize(process_str)
    for tkn in token_str:
        if len(tkn) > 1:
            clean_text.append(lemmatize_sentence(tkn))
    str_lema = " ".join(clean_text)
    clr_punct = word_tokenize(str_lema.lower())
    no_punct = [x for x in clr_punct if x not in stop]
    return no_punct


meta_tweet_1 = str(''.join(cluster_1))
meta_tweet_2 = str(''.join(cluster_2))
meta_tweet_3 = str(''.join(cluster_3))
meta_tweet_4 = str(''.join(cluster_4))
meta_tweet_5 = str(''.join(cluster_5))
meta_tweet_6 = str(''.join(cluster_6))
meta_tweet_7 = str(''.join(cluster_7))
meta_tweet_8 = str(''.join(cluster_8))
meta_tweet_9 = str(''.join(cluster_9))
meta_tweet_10 = str(''.join(cluster_10))
meta_tweet_11 = str(''.join(cluster_11))
meta_tweet_12 = str(''.join(cluster_12))

freq_words =[]
meta_1 = get_meta_tweet(meta_tweet_1)
freq_words.append(meta_1)
meta_2 = get_meta_tweet(meta_tweet_2)
freq_words.append(meta_2)
meta_3 = get_meta_tweet(meta_tweet_3)
freq_words.append(meta_3)
meta_4 = get_meta_tweet(meta_tweet_4)
freq_words.append(meta_4)
meta_5 = get_meta_tweet(meta_tweet_5)
freq_words.append(meta_5)
meta_6 = get_meta_tweet(meta_tweet_6)
freq_words.append(meta_6)
meta_7 = get_meta_tweet(meta_tweet_7)
freq_words.append(meta_7)
meta_8 = get_meta_tweet(meta_tweet_8)
freq_words.append(meta_8)
meta_9 = get_meta_tweet(meta_tweet_9)
freq_words.append(meta_9)
meta_10 = get_meta_tweet(meta_tweet_10)
freq_words.append(meta_10)
meta_11 = get_meta_tweet(meta_tweet_11)
freq_words.append(meta_11)
meta_12 = get_meta_tweet(meta_tweet_12)
freq_words.append(meta_12)

freq_list=[]
for x in freq_words:
    f = most_frequent(x)
    freq_list.append(f)

for y in freq_list:
    print(y)