import string 
import sklearn 
import pandas as pd
import numpy as np

import snowballstemmer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans

# Please note that running this program will take more than 1 hours on an average computer.

# This code does the following
# 1. combine title + abstract into "combined"
# 2. run clean_document for every row in the dataset 
# 3. run 3 different stemmers for the cleaned dataset, get 3 new datasets
# 4. for the 3 stemmed datasets, make a bagOfWords for each
# 5. Normalize the 3 bagOfWords 
# 6. Do the clustering 
# 7. Compare NMIs 
# 8. Pick the stemmer

# NOTE: All intermediate results will be stored in to their corresponding files.

# 1 a)
data_original = pd.read_csv("abstractdata5.csv", sep="#")
data_original["combined"] = data_original["title"] + " " + data_original["abstract"]
data_original.drop(["title", "abstract"], axis=1, inplace=True)

###### HELPERS
stops = list(stopwords.words('english'))
def clean_document(document, sw=stops):
    tokenized = word_tokenize(document.lower()) 
    filteredWords = []
    for w in tokenized:
        if (w not in sw) and (w not in string.punctuation):
            filteredWords.append(w)
    return filteredWords

def form_bag(data):
    bagOfWords = pd.DataFrame(dtype=int)
    for i, tokenList in enumerate(data.doc.values):
        print(i)
        for w in np.unique(np.array(tokenList)):
            bagOfWords.at[i,w] = int(list(tokenList).count(w))
    return bagOfWords

def normalize(data):
    res = data.copy()
    for rowNumber in list(data.index):
        doc = data.loc[rowNumber, :].dropna()
        numberOfWordsInDoc = np.sum(doc.values)
        res.at[rowNumber,:] = doc/numberOfWordsInDoc
    return res 

###### STEMMERS
snowball = snowballstemmer.stemmer('english')
porter = PorterStemmer()
lancaster = LancasterStemmer()

def stemSnowball(tokenList):
    return snowball.stemWords(tokenList)

def stemPorter(tokenList):
    res = np.zeros_like(tokenList)
    for i, token in enumerate(tokenList):
        res[i] = porter.stem(token)
    return res

def stemLancaster(tokenList):
    res = np.zeros_like(tokenList)
    for i, token in enumerate(tokenList):
        res[i] = lancaster.stem(token)
    return res


###### RUNNERS
def run_cleaning(data=data_original):
    data_cleaned = pd.DataFrame(columns=["doc"])
    for i, doc in enumerate(data.combined.values):
        cleaned = clean_document(doc)
        data_cleaned.at[i, "doc"] = cleaned
    data_cleaned.to_hdf("baseline_data_cleaned.h5", key="version1")
    print(data_cleaned.head())

def run_stemmers():
    data_cleaned = pd.read_hdf("baseline_data_cleaned.h5", key="version1")
    lancaster = pd.DataFrame(columns=["doc"])
    porter = pd.DataFrame(columns=["doc"])
    snowball = pd.DataFrame(columns=["doc"])
    for i, tokenList in enumerate(data_cleaned.doc.values):
        lancaster.at[i, "doc"] = stemLancaster(tokenList)
        porter.at[i, "doc"] = stemPorter(tokenList)
        snowball.at[i, "doc"] = stemSnowball(tokenList)
    lancaster.to_hdf("baseline_stemmers.h5", key="lancaster")
    porter.to_hdf("baseline_stemmers.h5", key="porter")
    snowball.to_hdf("baseline_stemmers.h5", key="snowball")
    print(lancaster.head())

    # This take the most time (approx. 1 hour with basic intel i7)
def run_bags():
    Lancaster
    data_lancaster = pd.read_hdf("baseline_stemmers.h5", key="lancaster")
    print("LANCASTER", data_lancaster.head())
    bag_lancaster = form_bag(data_lancaster)
    bag_lancaster.to_hdf("baseline_bags.h5", key="lancaster")
    # Porter
    data_porter = pd.read_hdf("baseline_stemmers.h5", key="porter")
    print("PORTER", data_porter.head())
    bag_porter = form_bag(data_porter)
    bag_porter.to_hdf("baseline_bags.h5", key="porter")
    # Snowball
    data_snowball = pd.read_hdf("baseline_stemmers.h5", key="snowball")
    print("SNOWBALL", data_snowball.head())
    bag_snowball = form_bag(data_snowball)
    bag_snowball.to_hdf("baseline_bags.h5", key="snowball")

    # does normalization and kmeans clustering for all of the three different stemming methods. 
    # Prints their NMI scores
def run_clustering():
    bag_lancaster = pd.read_hdf("baseline_bags.h5", key="lancaster")
    bag_porter = pd.read_hdf("baseline_bags.h5", key="porter")
    bag_snowball = pd.read_hdf("baseline_bags.h5", key="snowball")
    listOfBags = [(bag_porter, "porter"), 
                (bag_lancaster, "lancaster"), 
                (bag_snowball, "snowball")]
    K = 5
    true_labels = data_original["class"].values
    for bag in listOfBags:
        name = bag[1]
        print(name)
        data = bag[0]
        # 1c)
        X = normalize(data).fillna(0).values
        kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
        predLabels = kmeans.labels_
        # 1d)
        NMI = normalized_mutual_info_score(true_labels, predLabels, average_method="geometric")
        print("NMI: ", NMI)
        print()

# 1b)
# Run options, you can choose which operations to run.
RUN_CLEANING = True
RUN_STEMMING = True
RUN_BAGS = True
RUN_CLUSTERING = True

if RUN_CLEANING:
    print("BEGIN CLEANING")
    run_cleaning()
    print("END CLEANING")

if RUN_STEMMING:
    print("BEGIN STEMMING")
    run_stemmers()
    print("END STEMMING")

if RUN_BAGS:
    print("BEGIN BAGS")
    run_bags()
    print("END BAGS")

if RUN_CLUSTERING:
    print("BEGIN CLUSTERING")
    run_clustering()
    print("END CLUSTERING")


