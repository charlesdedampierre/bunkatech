import pandas as pd
import plotly
import warnings

from .semantics.extract_terms import extract_terms_df
from .semantics.indexer import indexer
from .semantics.get_embeddings import get_embeddings

from .hierarchical_clusters import hierarchical_clusters
from .hierarchical_clusters_label import hierarchical_clusters_label

from .visualization.make_bubble import make_bubble
from .visualization.sankey import make_sankey
from .visualization.topics_treemap import topics_treemap

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


def nested_topic_modeling(
    df: pd.DataFrame,
    text_var: str,
    index_var: str,
    sample_size: int = 500,
    sample_terms: int = 1000,
    embeddings_model: str = "tfidf",
):
    """[summary]

    Parameters
    ----------
    df : pd.DataFrame
    text_var : str
        Column with the text to extract topics from
    index_var : str
       index
    sample_size : int, optional
     by default 500

    Returns
    -------

        A treemap and a sankey diagram
    """

    df = df.sample(min(len(df), sample_size))
    df = df[df[text_var].notna()]  # cautious not to let any nan
    df = df.reset_index(drop=True)

    embeddings_reduced, _ = get_embeddings(
        df,
        field=text_var,
        model=embeddings_model,
        model_path="distiluse-base-multilingual-cased-v1",  # workds if sbert is chosen
    )

    # Create Nested clusters.
    df_emb = pd.DataFrame(embeddings_reduced)
    df_emb[index_var] = df[index_var]
    h_clusters = hierarchical_clusters(df_emb)

    print("Extract Terms...")
    terms = extract_terms_df(
        df,
        text_var=text_var,
        limit=4000,
        sample_size=sample_terms,
        ngs=True,  # ngrams
        ents=False,  # entities
        ncs=True,  # nouns
        drop_emoji=True,
        remove_punctuation=False,
        ngrams=(2, 2),
        include_pos=["NOUN", "PROPN", "ADJ"],
        include_types=["PERSON", "ORG"],
        language="en",
    )

    # Index with the list of original words
    df_terms = terms.copy()
    df_terms["text"] = df_terms["text"].apply(lambda x: x.split(" | "))
    df_terms = df_terms.explode("text").reset_index(drop=True)
    list_terms = df_terms["text"].tolist()

    # Index the extracted terms
    df_indexed = indexer(df[text_var].tolist(), list_terms, db_path=".")

    # get the Main form and the lemma
    df_indexed_full = pd.merge(df_indexed, df_terms, left_on="words", right_on="text")
    df_indexed_full = df_indexed_full[["docs", "lemma", "main form", "text"]].copy()

    # merge with terms extracted previously
    df_enrich = pd.merge(df, df_indexed_full, left_on=text_var, right_on="docs")
    df_enrich = df_enrich[[index_var, "main form"]].copy()

    # Merge clusters and original dataset
    h_clusters_label = pd.merge(h_clusters, df_enrich, on=index_var)
    h_clusters_label = h_clusters_label.rename(columns={"main form": "lemma"})
    h_clusters_names = hierarchical_clusters_label(h_clusters_label)

    # Make treemap
    treemap = topics_treemap(nested_topics=h_clusters_names)

    # Make Sankey Diagram
    sankey = make_sankey(h_clusters_names, field="Dataset", index_var=index_var)

    return treemap, sankey


if __name__ == "__main__":

    # Load Data
    df = pd.read_csv("./data/imdb_movies.csv", index_col=[0])
    treemap, sankey = nested_topic_modeling(
        df,
        text_var="description",
        index_var="imdb",
        sample_size=2000,
        sample_terms=3000,
        embeddings_model="tfidf",
    )
    plotly.offline.plot(sankey, auto_open=True, filename="saved_graph/sankey.html")
    plotly.offline.plot(treemap, auto_open=True, filename="saved_graph/treemap.html")

    plotly.offline.plot(sankey, auto_open=True, filename="saved_graph/sankey.html")
    plotly.offline.plot(treemap, auto_open=True, filename="saved_graph/treemap.html")
