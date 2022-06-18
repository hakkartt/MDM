from sentence_transformers import SentenceTransformer
from sklearn.cluster import SpectralClustering
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

import re
import matplotlib.pyplot as plt
import string 
import sklearn 
import pandas as pd
import numpy as np
import snowballstemmer
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer


data_original = pd.read_csv("abstractdata5.csv", sep="#")
data_original["combined"] = data_original["title"] + " " + data_original["abstract"]
data_original.drop(["title", "abstract"], axis=1, inplace=True)


embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus with example sentences
corpus = list(data_original.combined)
corpus_embeddings = embedder.encode(corpus)

# Perform kmean clustering
num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
predLabels = clustering_model.labels_
true_labels = data_original["class"].values

nmi = normalized_mutual_info_score(true_labels, predLabels, average_method="geometric")
print(nmi)

# clustered_sentences = [[] for i in range(num_clusters)]
# for sentence_id, cluster_id in enumerate(cluster_assignment):
#     clustered_sentences[cluster_id].append(corpus[sentence_id])

# for i, cluster in enumerate(clustered_sentences):
#     print("Cluster ", i+1)
#     print(cluster)
#     print("")