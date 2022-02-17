import pandas as pd
import warnings
from sklearn.cluster import KMeans
from .semantics.extract_terms import extract_terms_df
from .semantics.indexer import indexer
from .semantics.get_embeddings import get_embeddings
from .visualization.make_bubble import make_bubble
from .specificity import specificity
from sklearn.cluster import OPTICS

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


def topic_modeling(
    docs: list,
    topic_number=20,
    top_terms=20,
    ngrams=(1, 2),
    ents=False,
    sample_terms=2000,
    model="tfidf",
    language="en",
):

    df = pd.DataFrame(docs, columns=["text_var"])

    print("Create Embeddings & Umap...")
    # Embeddings & UMAP
    embeddings_reduced, _ = get_embeddings(
        df,
        field="text_var",
        model=model,
        model_path="distiluse-base-multilingual-cased-v1",  # workds if sbert is chosen
    )

    # clusters
    print("Create Clusters...")

    """df["cluster"] = (
        OPTICS(min_cluster_size=1 / 20).fit(embeddings_reduced).labels_.astype(str)
    )"""

    df["cluster"] = (
        KMeans(n_clusters=topic_number).fit(embeddings_reduced).labels_.astype(str)
    )

    topic_size = df.groupby("cluster").count().reset_index()
    topic_size.columns = ["cluster", "topic_size"]

    print("Extract Terms...")
    # Extract Terms
    terms = extract_terms_df(
        df,
        text_var="text_var",
        limit=2000,
        sample_size=sample_terms,
        ngs=True,  # ngrams
        ents=ents,  # entities
        ncs=False,  # nouns
        drop_emoji=True,
        remove_punctuation=False,
        ngrams=ngrams,
        include_pos=["NOUN", "PROPN", "ADJ"],
        include_types=["PERSON", "ORG"],
        language=language,
    )

    # Index with the list of original words
    df_terms = terms.copy()
    df_terms["text"] = df_terms["text"].apply(lambda x: x.split(" | "))
    df_terms = df_terms.explode("text").reset_index(drop=True)
    list_terms = df_terms["text"].tolist()

    df_indexed = indexer(df["text_var"].tolist(), list_terms, db_path=".")
    # get the Main form and the lemma
    df_indexed_full = pd.merge(df_indexed, df_terms, left_on="words", right_on="text")
    df_indexed_full = df_indexed_full[["docs", "lemma", "main form", "text"]].copy()

    data = pd.merge(df_indexed_full, df, left_on="docs", right_on="text_var")
    data = data.drop("docs", axis=1)

    _, _, edge = specificity(data, X="cluster", Y="main form", Z=None, top_n=top_terms)

    topics = (
        edge.groupby("cluster")["main form"]
        .apply(lambda x: " | ".join(x))
        .reset_index()
    )

    final = pd.merge(topics, topic_size, on="cluster")
    final = final.sort_values("topic_size", ascending=False)
    final = final.reset_index(drop=True)

    return final
