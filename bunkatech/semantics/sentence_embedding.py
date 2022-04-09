import pandas as pd
import pickle
from tqdm import tqdm


def from_dict_to_frame(indexed_dict):
    data = {k: [v] for k, v in indexed_dict.items()}
    df = pd.DataFrame.from_dict(data).T
    df.columns = ["text"]
    df = df.explode("text")
    return df


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_sentence_embeddings(terms_embeddings: pd.DataFrame, df_terms_indexed: dict):

    final = pd.merge(
        terms_embeddings, df_terms_indexed, right_index=True, left_index=True
    ).reset_index()
    final = final.drop("text", axis=1)
    final = final.groupby("index").mean()
    return final


if __name__ == "__main__":
    path = "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/demo_day/content/"
    terms_embeddings = pd.read_csv(
        path + "/terms_embeddings.csv",
        index_col=[0],
    )
    with open(path + "/terms_indexed.pickle", "rb") as handle:
        terms_indexed = pickle.load(handle)

    df_terms_indexed = from_dict_to_frame(terms_indexed)
    df_terms_indexed = df_terms_indexed.reset_index().set_index("text")

    keys = list(terms_indexed.keys())
    chunks_list = list(chunks(keys, 5000))

    final_emb = pd.DataFrame()
    for list_ids in tqdm(chunks_list, total=len(chunks_list)):
        terms_indexed_filtered = df_terms_indexed[
            df_terms_indexed["index"].isin(list_ids)
        ]
        semb = get_sentence_embeddings(terms_embeddings, terms_indexed_filtered)
        final_emb = final_emb.append(semb)

    final_emb.to_csv(path + "/semb.csv")
