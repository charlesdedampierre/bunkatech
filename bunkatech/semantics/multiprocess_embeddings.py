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

<<<<<<< HEAD
if __name__ == "__main__":
    path = "/Users/charlesdedampierre/Desktop/SciencePo Projects/shaping-ai/search_test"
    df_index = pd.read_csv(
        path + "/df_index.csv",
        index_col=[0],
    )

    docs = df_index.reset_index()["text"].drop_duplicates().dropna().to_list()
=======

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

    #path = "/Users/charlesdedampierre/Desktop/wiki_movie_plots_deduped.csv"

    data = pd.read_csv(path + '/SHAI-CORPUS-MODEL-ALL-R2.csv', sep = ';')
    data = data[data['content'].notna()]


    data = pd.read_csv('/Users/charlesdedampierre/Desktop/ENS Projects/humility/extended_dataset/extended_training_dataset.csv')

    docs = data['body'].tolist()
    index = data['id'].tolist()
    #data = data.reset_index()
    #docs = data['content'].tolist()
    #index = data['unique_id'].tolist()

    print(data.columns)
>>>>>>> staging

    with Pool(8) as p:
        res = list(tqdm(p.imap(embed, docs), total=len(docs)))

<<<<<<< HEAD
    df_res = pd.DataFrame(res, index=docs)
    df_res.to_csv(path + "/terms_embeddings_sbert.csv")
=======
    df_res = pd.DataFrame(res, index=index)
    df_res.to_csv('/Users/charlesdedampierre/Desktop/ENS Projects/humility/extended_dataset/bert_embed.csv')


    """
    df_res.to_csv('/Users/charlesdedampierre/Desktop/bunkatech/wiki_emb.csv')

   
 
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
        pickle.dump(handle, protocol=pickle.HIGHEST_PROTOCOL)

        """
    
    
>>>>>>> staging
