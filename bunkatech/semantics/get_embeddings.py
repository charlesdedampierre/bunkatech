import pandas as pd
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


def get_embeddings(
    data: pd.DataFrame,
    field: str,
    model_path: str,
    model="sbert",
    reduction_size: int = 5,
):

    print("Embeddings..")
    if model == "sbert":
        model = SentenceTransformer(model_path)
        docs = list(data[field])
        print("Territory embedding..")
        emb = model.encode(docs, show_progress_bar=True)

    elif model == "tfidf":
        model = TfidfVectorizer(max_features=20000)
        sentences = list(data[field])
        emb = model.fit_transform(sentences)
    else:
        raise ValueError(
            f"{model} is not a right embeddings_model. Please chose between 'tfidf' or 'sbert'"
        )

    print("Reducing the vectors..")
    red = umap.UMAP(
        n_components=reduction_size, n_neighbors=10, metric="cosine", verbose=True
    )

    emb_red = red.fit_transform(emb)

    return emb_red, emb
