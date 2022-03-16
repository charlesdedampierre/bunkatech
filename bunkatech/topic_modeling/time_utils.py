import numpy as np
from sklearn.metrics import pairwise_distances


def cosine_distance_exponential_time_decay(bert_vectors, timestamps, temperature=1):
    """Compute embeddings with temporal adjustment"""

    # compute distances between embeddings
    cosine_distance_matrix = pairwise_distances(bert_vectors, metric="cosine")
    # We use l2 since the implementation for pairwise_distances() is much faster than our custom (x-y)²

    # compute distances between timestamps
    timematrix = pairwise_distances(np.array(timestamps).reshape(-1, 1), metric="l2")
    timematrix_renorm = timematrix / np.max(timematrix)
    exp_timematrix = np.exp(-timematrix_renorm / temperature)

    # Join the two embeddings
    final_emb = cosine_distance_matrix * exp_timematrix

    return final_emb
