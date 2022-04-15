import pandas as pd
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool
from tqdm import tqdm
import umap
import numpy as np
import glob
import pickle

model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
model = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')

def concat_files(path):
    files = glob.glob(path + "/*")
    data = pd.DataFrame()
    for file in files:
        dat = pd.read_csv(file, index_col=[0])
        data = data.append(dat)
    return data


# terms_indexed.reset_index().groupby('unique_id')['text'].apply(list).to_dict()


def from_dict_to_frame(indexed_dict):
    data = {k: [v] for k, v in indexed_dict.items()}
    df = pd.DataFrame.from_dict(data).T
    df.columns = ["text"]
    df = df.explode("text")
    return df


def embed(text):
    text = str(text)
    emb = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
    return emb


"""def multi_embedding(docs):

    docs_embedding_model = "distiluse-base-multilingual-cased-v1"
    model = SentenceTransformer(docs_embedding_model)
    pool = model.start_multi_process_pool()
    docs_embeddings = model.encode_multi_process(docs, pool, show_progress_bar=True)
    model.stop_multi_process_pool(pool)

    return docs_embeddings
"""


def get_sentence_embeddings(
    terms_embeddings: pd.DataFrame, df_terms_indexed: dict, index_var="unique_id"
):

    final = pd.merge(
        terms_embeddings, df_terms_indexed, right_index=True, left_index=True
    ).reset_index()
    final = final.drop("text", axis=1)
    final = final.groupby(index_var).mean()
    return final


if __name__ == "__main__":
    path = "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/demo_day"

    path = "/Users/charlesdedampierre/Desktop/wiki_movie_plots_deduped.csv"

    data = pd.read_csv(path)
    data = data.reset_index()
    docs = data['Plot'].tolist()
    index = data['index'].tolist()
    with Pool(8) as p:
        res = list(tqdm(p.imap(embed, docs), total=len(docs)))

    df_res = pd.DataFrame(res, index=index)
    df_res.to_csv('/Users/charlesdedampierre/Desktop/bunkatech/wiki_emb.csv')

   
    """
    all_terms = pd.read_csv(path + "/content/terms.csv")
    chunks_terms = np.array_split(all_terms, 10)
    x = 0
    for terms in chunks_terms:
        print(x)
        docs = list(terms["text"])

        with Pool(8) as p:
            res = list(tqdm(p.imap(embed, docs), total=len(docs)))

        df_res = pd.DataFrame(res, index=docs)
        df_res.index.name = "text"

    
        
        red = umap.UMAP(n_components=5).fit()
        res_red = umap.UMAP(n_components=5).fit_transform(df_res)
        df_res_red = pd.DataFrame(res_red, index=docs)
        df_res_red.index.name = "text

        df_res.to_csv(
            path + f"/content/terms_embeddings/terms_embeddings_sbert_{x}.csv"
        )
        x += 1

        with open(
        "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/demo_day/content/umap_model.pickle",
        "wb",
        ) as handle:
        pickle.dump(handle, protocol=pickle.HIGHEST_PROTOCOL)"""
    