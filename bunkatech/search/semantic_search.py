import pandas as pd
from tqdm import tqdm
import numpy as np
from fuzzywuzzy import fuzz, process

tqdm.pandas()
from multiprocessing import Pool

multiprocessing_pools = 8

import textacy
import textacy.preprocessing
import textacy.representations
import textacy.tm

from functools import partial
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


queries = ["danger pour l'homme"]


def semantic_search(
    queries: list,
    embeddings_sbert: pd.DataFrame,
    model_path="/Users/charlesdedampierre/Desktop/transformer_models/distiluse-base-multilingual-cased-v1",
):
    model = SentenceTransformer(model_path)
    full_cos = pd.DataFrame()
    for query in queries:

        query_emb = model.encode(query, show_progress_bar=False)

        # Get the closest embeddings
        cos_scores = util.pytorch_cos_sim(
            query_emb, np.array(embeddings_sbert).astype(np.float32)
        )
        df_cos_scores = pd.DataFrame(
            np.array(cos_scores), columns=embeddings_sbert.index, index=["cos_score"]
        ).T
        df_cos_scores = df_cos_scores.sort_values("cos_score", ascending=False).head(10)

        df_cos_scores["query"] = query
        full_cos = full_cos.append(df_cos_scores)
        full_cos = full_cos.drop_duplicates()

    full_cos = full_cos.sort_values("cos_score", ascending=False)
    full_cos["cos_score"] = np.exp(full_cos["cos_score"])

    return full_cos


def exact_search(query, terms, top_n=10):

    terms["ratio"] = terms["text"].progress_apply(
        lambda x: fuzz.token_sort_ratio(query, x)
    )
    res = terms.sort_values("ratio", ascending=False).head(top_n).reset_index(drop=True)

    return res


def pipeline_search(
    queries,
    index_var,
    embeddings_sbert: pd.DataFrame,
    terms_indexed: pd.DataFrame,
    data: pd.DataFrame,
    search_type="semantic_search",
):

    if search_type is "semantic_search":
        full_cos = semantic_search(queries, embeddings_sbert)

        final = pd.merge(full_cos, terms_indexed, left_index=True, right_index=True)
        final = final.sort_values([index_var, "cos_score"], ascending=(False, False))
        final = final.reset_index()

        func = lambda x: " | ".join(x)
        output = final.groupby(index_var).agg(
            score_total=("cos_score", "sum"), text_list=("index", func)
        )

        res = pd.merge(output, data, left_index=True, right_index=True)
        res = res.sort_values("score_total", ascending=False)

    return res
