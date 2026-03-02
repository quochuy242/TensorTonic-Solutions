import numpy as np
from collections import Counter
from itertools import chain
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, terms).
    """
    # Write code here

    # Handle empty documents
    if not documents:
        return np.zeros((0, 0)), []

    tokenized_documents = [
        doc.lower().split() if doc else []
        for doc in documents
    ]
    
    terms = sorted(set(
        term 
        for doc in tokenized_documents 
        for term in doc
    ))

    n_docs = len(documents)
    n_vocab = len(terms)

    if n_vocab == 0:
        return np.zeros((n_docs, 0)), terms

    vocab_index = {
        term: i for i, term in enumerate(terms)
    }

    # df
    df = Counter()
    for doc in tokenized_documents:
        df.update(set(doc))

    # idf
    idf = {term: np.log(n_docs / df[term]) for term in terms}

    # tfidf
    tfidf = np.zeros((n_docs, n_vocab), dtype=float)
    for d_idx, doc in enumerate(tokenized_documents):
        if not doc:
            continue 

        counts = Counter(doc)
        doc_len = len(doc)

        for term, cnt in counts.items():
            t_idx = vocab_index[term]
            tf = cnt / doc_len 

            tfidf[d_idx, t_idx] = tf * idf[term]
            
    return tfidf, terms 
    