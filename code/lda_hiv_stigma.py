import argparse
import random
import sys
sys.path.append("../")
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import string
from tqdm import trange
import nltk
import math
import warnings
import pickle
from itertools import combinations
import os
import gensim

from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download('stopwords')
nltk.download('wordnet')

warnings.simplefilter(action='ignore', category=FutureWarning)


'''
Load and Preprocessing of notes
'''
hiv= pd.read_csv('LDA_hiv1.csv')
print(hiv.shape)
hiv['note_text'] = hiv['note_text'].astype(str)
doc_complete = hiv['note_text'].tolist()

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punct_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punct_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]    
print('\n\nCleaned Data\n\n')
print(doc_clean[:1])

'''
Modeling
'''
# Term dictionary of Corpus 
dictionary = corpora.Dictionary(doc_clean)
dictionary.filter_extremes(no_below = 5, no_above= .9)
# Document Term Matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


def jaccard_similarity(set1, set2):
        return len(set1 & set2) / len(set1 | set2)

def compute_avg_jaccard(ldamodel, top_n=10):
    top_words_per_topic = [
        set([word for word, _ in ldamodel.show_topic(tid, top_n)])
        for tid in range(ldamodel.num_topics)
    ]
    similarities = [
        jaccard_similarity(t1, t2)
        for t1, t2 in combinations(top_words_per_topic, 2)
    ]
    return np.mean(similarities)

def calculate_topic_diversity(ldamodel, top_n=25):
            topics = ldamodel.show_topics(num_topics=-1, num_words=top_n, formatted=False)
            word_list = [word for topic in topics for word, _ in topic[1]]
            unique_words = set(word_list)
            return len(unique_words) / (ldamodel.num_topics * top_n)

results = []

for iter in range (10):
    print(f"Iteration {iter}")

    for top_k in range (5, 35, 5):
        print(f"Training LDA with {top_k} topics...")

        Lda = gensim.models.ldamodel.LdaModel

        ldamodel = gensim.models.LdaModel(
                corpus=doc_term_matrix,
                num_topics=top_k,
                id2word=dictionary,
                passes=50,
                random_state=iter
            )
        print(ldamodel.print_topics(num_topics=top_k, num_words=10))

        #Evaluation-1
        coherence_model = CoherenceModel(
              model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v'
              )
        coherence = coherence_model.get_coherence()

        #Evaluation-2
        jaccard = compute_avg_jaccard(ldamodel)
        
        #Evaluation-3
        diversity = calculate_topic_diversity(ldamodel, top_n=25)

        print(f"Coherence: {coherence:.4f}, Avg Jaccard: {jaccard:.4f}, Diversity: {diversity:.4f}")
        print("-" * 50)

        results.append({
            "iteration": iter,
            "num_topics": top_k,
            "coherence_score": coherence,
            "avg_jaccard_similarity": jaccard,
            "topic_diversity": diversity
        })

df_lda_results = pd.DataFrame(results)
print(df_lda_results)

df_lda_results.to_csv('output.csv', index=False)  
