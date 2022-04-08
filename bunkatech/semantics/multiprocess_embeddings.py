import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from tqdm import tqdm

model = SentenceTransformer("distiluse-base-multilingual-cased-v1")


def embed(text):
    emb = model.encode(text, show_progress_bar=False)
    return emb


"""def multi_embedding(docs):

    docs_embedding_model = "distiluse-base-multilingual-cased-v1"
    model = SentenceTransformer(docs_embedding_model)
    pool = model.start_multi_process_pool()
    docs_embeddings = model.encode_multi_process(docs, pool, show_progress_bar=True)
    model.stop_multi_process_pool(pool)

    return docs_embeddings
"""

if __name__ == "__main__":
    path = "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/search_test"
    df_index = pd.read_csv(
        path + "/df_index.csv",
        index_col=[0],
    )

    docs = df_index.reset_index()["text"].drop_duplicates().dropna().to_list()

    with Pool(8) as p:
        res = list(tqdm(p.imap(embed, docs), total=len(docs)))

    df_res = pd.DataFrame(res, index=docs)
    df_res.to_csv(path + "/terms_embeddings_sbert.csv")
