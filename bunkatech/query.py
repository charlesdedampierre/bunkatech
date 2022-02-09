import pandas as pd
from sentence_transformers import util, SentenceTransformer
import numpy as np


def make_query(
    data: pd.DataFrame,
    model: SentenceTransformer,
    corpus_embeddings: np.array,
    query: list,
    data_id: str = "bindex",
):
    """Compare the embedding of a query to the global embedding end merge the results
    with the index of data

    Args:
        data (pd.DataFrame): data of the observation
        corpus_embeddings (np.array): emebddings of the index in the data
        query (list): query to input
        data_id (str, optional): columns of the index to "bindex".

    Returns:
        [type]: a Dataframe with the index and the score of the query. The higher, the similar
    """

    top_n = int(round(len(data) / 100, 0))
    # top_k = min(top_n, len(corpus))

    # Compute the scores
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(
        query_embedding, corpus_embeddings.astype(np.float32)
    )[0]

    score_list = []
    index_list = []

    x = 0
    for score in cos_scores:
        index_list.append(data[data_id].iloc[x])
        score_list.append(round(score.item(), 2))
        x += 1

    # Make data
    d = {"bindex": index_list, "score": score_list}

    final = pd.DataFrame(data=d)
    final["query"] = " | ".join(query)
    final = final.sort_values("score", ascending=False)
    final = final.head(top_n)
    final = final.reset_index(drop=True)

    return final


query = ["Aliens are everywhere"]
query = None

if query is not None:
    full_emb = pd.read_csv(storage_path + "/embeddings/full_emb.csv", index_col=[0])
    full_emb = full_emb.drop("imdb", axis=1)
    full_emb = full_emb.values
    data = data.rename(columns={"imdb": "bindex"})

    model = SentenceTransformer(model_path)
    query_result = make_query(data, model, full_emb, query, data_id="bindex")
else:
    query_result = None
