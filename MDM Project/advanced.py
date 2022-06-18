import re
import matplotlib.pyplot as plt
import string 
import sklearn 
import pandas as pd
import numpy as np
import snowballstemmer
from sklearn.cluster import SpectralClustering
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
# from nltk.corpus import words
# sample="ASD KISSA KALA Koira KaLa (CONSORT)-AI"
# pd.set_option('precision', 0)


# Please note that running this program will take more than 1 hours on an average computer.

# This code does the following
# 1. combine title + abstract into "combined"
# 2. run clean_document for every row in the dataset 
# 3. run only Lancaster stemmer for the cleaned dataset, get a new datasets
# 4. for the stemmed dataset, make a bagOfWords 
# 5. filter out (trim) words that occur only 1, 5, or 10 times
# 6. Normalize the bagOfWords 
# 7. Do SVD (LSA)
# 8. Do Spectral clustering 
# 9. Optimize amount of dimensions (LSA components) and gamma
# 7. Compare NMIs 
# 8. Pick the best parameter setting

# NOTE: All intermediate results will be stored in to their corresponding files.


data_original = pd.read_csv("abstractdata5.csv", sep="#")
data_original["combined"] = data_original["title"] + " " + data_original["abstract"]
data_original.drop(["title", "abstract"], axis=1, inplace=True)


###### STEMMERS & LEMMERS
lancaster = LancasterStemmer()
snowball = snowballstemmer.stemmer('english')
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()

def stemLancaster(tokenList):
    res = np.zeros_like(tokenList)
    for i, token in enumerate(tokenList):
        res[i] = lancaster.stem(token)
    return res

def stemSnowball(tokenList):
    return snowball.stemWords(tokenList)

def stemPorter(tokenList):
    res = np.zeros_like(tokenList)
    for i, token in enumerate(tokenList):
        res[i] = porter.stem(token)
    return res

def lemmer(tokenList):
    res = np.zeros_like(tokenList)
    for i, token in enumerate(tokenList):
        res[i] = lemmatizer.lemmatize(token)
    return res

###### HELPERS
stops = list(stopwords.words('english'))
    # Does lowercasing and removes hyphens, stopwords, punctuation, and numbers
def clean_document(document, sw=stops):
    reg = re.compile(r'^[\w-]+$')
    lower = document.lower() # lowercasing
    trimmed = lower.replace("-", " ")
    tokenized = word_tokenize(trimmed) 
    filteredWords = []
    for w in tokenized:
        if (w not in sw) and (w not in string.punctuation) and not (w.isnumeric()) and reg.match(w):
            filteredWords.append(w)
    return filteredWords

def form_bag(data):
    bagOfWords = pd.DataFrame(dtype=int)
    for i, tokenList in enumerate(data.doc.values):
        print(i)
        for w in np.unique(np.array(tokenList)):
            bagOfWords.at[i,w] = int(list(tokenList).count(w))
    return bagOfWords

def normalize_basic(data):
    res = data.copy()
    for rowNumber in list(data.index):
        doc = data.loc[rowNumber, :].dropna()
        numberOfWordsInDoc = np.sum(doc.values)
        res.at[rowNumber,:] = doc/numberOfWordsInDoc
    return res 

def normalize_tfidf_weighting(data):
    totalDocs = data.shape[0]
    wordOccurrences = data.count()
    res = data.copy()
    for rowNumber in list(data.index):
        doc = data.loc[rowNumber, :].dropna()
        numberOfWordsInDoc = np.sum(doc.values)
        for word in doc.index:
            wordOccurence = wordOccurrences[word]
            wordFrequencyInDoc = doc[word]
            res.at[rowNumber,word] = (wordFrequencyInDoc/numberOfWordsInDoc) * np.log(totalDocs/wordOccurence)
    return res 

def trimBag(bag, threshold):
    res = bag.copy()
    counts = bag.count()
    for i, word in enumerate(counts.index):
        if counts[word] <= threshold:
            print(i)
            res = res.drop(columns=[word])
    print(res.count().min())
    return res

def doc_bigrams(tokenList, ngrams):
    skip = False
    newList = []
    for j in range(0, len(tokenList)-1):
        if skip: 
            skip = False
        else:
            firstWord = tokenList[j]
            pair = firstWord + " " + tokenList[j+1]
            if pair in ngrams:
                newList.append(pair)
                skip = True
            else:
                newList.append(firstWord)
    if not skip:
        newList.append(tokenList[-1])
    return newList

def form_ngram(data, ngram_range=(2,3)):
    c_vec = CountVectorizer(ngram_range=ngram_range)
    strings = []
    for i, tokenList in enumerate(data.doc.values):
        strings.append((" ").join(tokenList))
    ngrams = c_vec.fit_transform(strings)
    # count frequency of ngrams
    count_values = ngrams.toarray().sum(axis=0)
    # list of ngrams
    vocab = c_vec.vocabulary_
    df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
                ).rename(columns={0: 'frequency', 1:'ngram'})
    ngram = df_ngram[df_ngram.frequency > 100]
    return list(ngram.ngram)


########################### RUNNERS

def run_cleaning(data=data_original):
    data_cleaned = pd.DataFrame(columns=["doc"])
    for i, doc in enumerate(data.combined.values):
        cleaned = clean_document(doc)
        data_cleaned.at[i, "doc"] = cleaned
    print(data_cleaned.head())
    data_cleaned.to_hdf("advanced_data_cleaned.h5", key="version1")

def run_stemming():
    data_cleaned = pd.read_hdf("advanced_data_cleaned.h5", key="version1")
    lancaster = pd.DataFrame(columns=["doc"])
    for i, tokenList in enumerate(data_cleaned.doc.values):
        lancaster.at[i, "doc"] = stemLancaster(tokenList)
    lancaster.to_hdf("advanced_stemmers.h5", key="lancaster")
    print(lancaster.head())

def run_bags():
    data_lancaster = pd.read_hdf("advanced_stemmers.h5", key="lancaster")
    bag_lancaster = form_bag(data_lancaster)
    bag_lancaster.to_hdf("advanced_bags.h5", key="lancaster")
    print(bag_lancaster.head())

    # Filters our words that have less occurrences than some threshold
def run_trimming():
    bag_lancaster = pd.read_hdf("advanced_bags.h5", key="lancaster")
    thresholds = [1, 10, 5]
    for thresh in thresholds:
        trimmed = trimBag(bag_lancaster, thresh)
        trimmed.to_hdf("advanced_trimmed_bags.h5", key=str(thresh))
    
def run_normalizing():
    lancaster_trimmed_1 = pd.read_hdf("advanced_trimmed_bags.h5", key=str(1))
    lancaster_trimmed_5 = pd.read_hdf("advanced_trimmed_bags.h5", key=str(5))
    lancaster_trimmed_10 = pd.read_hdf("advanced_trimmed_bags.h5", key=str(10))
    bags = [(lancaster_trimmed_1, 1), (lancaster_trimmed_10, 10), (lancaster_trimmed_5, 5)]
    for bag in bags:
        data = bag[0]
        thresh = bag[1]
        trimmed = normalize_tfidf_weighting(data)
        trimmed.to_hdf("advanced_normalized_bags.h5", key=str(thresh))

def run_dimred_and_clustering():
    lancaster_1 = pd.read_hdf("advanced_normalized_bags.h5", key=str(1))
    # lancaster_5 = pd.read_hdf("advanced_normalized_bags.h5", key=str(5))
    # lancaster_10 = pd.read_hdf("advanced_normalized_bags.h5", key=str(10))
    # datasets = [(lancaster_1, 1), (lancaster_5, 5), (lancaster_10, 10)]
    X = lancaster_1.fillna(0).values
    n_components = np.arange(80, 135, 1)
    res = []
    bestNMI = 0
    bestN = 0
    for n in n_components:
        print(n)
        svd = TruncatedSVD(n_components=n, algorithm="arpack")
        X_transformed = svd.fit_transform(X)
        spectral = SpectralClustering(n_clusters=K,
                                    gamma=1,
                                    assign_labels='discretize'                    
                                    ).fit(X_transformed)                                
        spectralPred = spectral.labels_
        nmi = normalized_mutual_info_score(true_labels, spectralPred, average_method="geometric")
        print("NMI ", nmi)
        res.append(nmi)
        if nmi > bestNMI:
            bestNMI = nmi
            bestN = n
    print(bestNMI, bestN)
    plt.plot(list(n_components), res, "-")
    plt.xlabel("n_components")
    plt.ylabel("NMI")
    plt.title("Spectral, Lancaster, LSA")
    plt.figtext(.8, .8, "n = {} \nNMI = {}".format(bestN, bestNMI))
    plt.savefig("advaced_nmi_plot_n_components_small.png")

def run_gamma():
    lancaster_1 = pd.read_hdf("advanced_normalized_bags.h5", key=str(1))
    X = lancaster_1.fillna(0).values
    gammas = np.arange(0.9, 1.3, 0.01)
    # gammas = [0.0001,0.001,0.01,0.1,1,10,20]
    res = []
    bestNMI = 0
    bestGamma = 0
    for gamma in gammas:
        print(gamma)
        svd = TruncatedSVD(n_components=130, algorithm="arpack")
        X_transformed = svd.fit_transform(X)
        spectral = SpectralClustering(n_clusters=K,
                                    eigen_solver="arpack",
                                    gamma=gamma,
                                    assign_labels='discretize'                    
                                    ).fit(X_transformed)                                
        predLabels = spectral.labels_
        nmi = normalized_mutual_info_score(true_labels, predLabels, average_method="geometric")
        res.append(nmi)
        if nmi > bestNMI:
            bestNMI = nmi
            bestGamma = gamma
        print(nmi)
    print(bestNMI, bestGamma)
    plt.plot(list(gammas), res, "-")
    plt.xlabel("gamma")
    plt.ylabel("NMI")
    plt.title("Spectral, Lancaster, LSA")
    plt.figtext(.8, .8, "gamma = {} \nNMI = {}".format(bestGamma, bestNMI))
    plt.savefig("advaced_gamma_plot.png")

def get_nmi_mean():
    lancaster_1 = pd.read_hdf("advanced_normalized_bags.h5", key=str(1))
    X = lancaster_1.fillna(0).values
    res = []
    for i in range(0,101):
        svd = TruncatedSVD(n_components=130, algorithm="arpack")
        X_transformed = svd.fit_transform(X)
        spectral = SpectralClustering(n_clusters=K,
                                    eigen_solver="arpack",
                                    gamma=1.03,
                                    assign_labels='discretize'                    
                                    ).fit(X_transformed)                                
        predLabels = spectral.labels_
        nmi = normalized_mutual_info_score(true_labels, predLabels, average_method="geometric")
        res.append(nmi)
        print(i, nmi)
    nmi_mean = np.mean(res)
    nmi_max = np.max(res)
    nmi_min = np.min(res)
    print(nmi_mean)
    plt.plot(res)
    plt.axhline(y=nmi_min, color="r", linestyle="dashed", label="min={}".format((np.round(nmi_min, 3))))
    plt.axhline(y=nmi_max, color="g", linestyle="dashed", label="max={}".format((np.round(nmi_max, 3))))
    plt.axhline(y=nmi_mean, color="b", linestyle="dashed", label="mean={}".format((np.round(nmi_mean, 3))))
    plt.legend(loc="lower left")
    plt.xlabel("iterations")
    plt.ylabel("NMI")
    plt.title("Spectral, Lancaster, LSA, dim=130, gamma=1.03")
    plt.savefig("NMI_mean.png")

def get_clusters():
    lancaster_1 = pd.read_hdf("advanced_normalized_bags.h5", key=str(1))
    X = lancaster_1.fillna(0).values
    svd = TruncatedSVD(n_components=130, algorithm="arpack")
    X_transformed = svd.fit_transform(X)
    clustering_model = SpectralClustering(n_clusters=K,
                                    eigen_solver="arpack",
                                    gamma=1.03,
                                    assign_labels='discretize'                    
                                    ).fit(X_transformed) 
    cluster_assignment = clustering_model.labels_
    lancaster_1.at[:, "label"] = cluster_assignment


    for i in range(0, 5):
        print("Cluster", i)
        cluster = lancaster_1[lancaster_1.label == i]
        frequent_words = cluster.count()
        print(frequent_words.sort_values(ascending=False).head(20))
        print()


# Run options, you can choose which operations to run.
RUN_CLEANING = True
RUN_STEMMING = True
RUN_BAGS = True
RUN_TRIMMING = True
RUN_NORMALIZING = True
RUN_DIMRED_AND_CLUSTERING = True
RUN_GAMMA = True
RUN_NMI_MEAN = True
GET_CLUSTERS = True

# Constans for runners
true_labels = data_original["class"].values
K = 5

if RUN_CLEANING:
    print("BEGIN CLEANING")
    run_cleaning()
    print("END CLEANING")

if RUN_STEMMING:
    print("BEGIN STEMMING & LEMMING")
    run_stemming()
    print("END STEMMING & LEMMING")

if RUN_BAGS:
    print("BEGIN BAGS")
    run_bags()
    print("END BAGS")

if RUN_TRIMMING:
    print("BEGIN TRIMMING")
    run_trimming()
    print("END TRIMMING")

if RUN_NORMALIZING:
    print("BEGIN NORMALIZING")
    run_normalizing()
    print("END NORMALIZING")

if RUN_DIMRED_AND_CLUSTERING:
    print("BEGIN DIMENSION REDUCTION")
    run_dimred_and_clustering()
    print("END DIMENSION REDUCTION")

if RUN_GAMMA:
    print("BEGIN GAMMA OPTIMIZATION")
    run_gamma()
    print("END GAMMA OPTIMIZATION")

if RUN_NMI_MEAN:
    print("BEGIN NMI MEAN")
    get_nmi_mean()
    print("END NMI MEAN")

if GET_CLUSTERS:
    print("BEGIN CLUSTERING RESULTS")
    get_clusters()
    print("END CLUSTERING RESULTS")