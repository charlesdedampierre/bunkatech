import pandas as pd
import numpy as np

import plotly
import plotly.express as px
import warnings
import umap

from .semantics.extract_terms import extract_terms_df
from .semantics.indexer import indexer
from .semantics.get_embeddings import get_embeddings

from .hierarchical_clusters import hierarchical_clusters
from .hierarchical_clusters_label import hierarchical_clusters_label

from .visualization.make_bubble import wrap_by_word
from .visualization.sankey import make_sankey
from .visualization.topics_treemap import topics_treemap
from .visualization.topics_sunburst import topics_sunburst

from .networks.centroids import find_centroids


warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


class nested_topic_modeling:
    def __init__(self) -> None:
        pass

    def fit(
        self,
        df: pd.DataFrame,
        text_var: str,
        index_var: str,
        sample_size: int = 500,
        sample_terms: int = 1000,
        embeddings_model: str = "tfidf",
        ngrams=(1, 2),
        ents=False,
        language="en",
        db_path=".",
    ):

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

        terms = extract_terms_df(
            df,
            text_var=text_var,
            limit=4000,
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

        # Index the extracted terms
        df_indexed = indexer(df[text_var].tolist(), list_terms, db_path=db_path)

        # get the Main form and the lemma
        df_indexed_full = pd.merge(
            df_indexed, df_terms, left_on="words", right_on="text"
        )
        df_indexed_full = df_indexed_full[["docs", "lemma", "main form", "text"]].copy()

        # merge with terms extracted previously
        df_enrich = pd.merge(df, df_indexed_full, left_on=text_var, right_on="docs")
        df_enrich = df_enrich[[index_var, "main form"]].copy()

        # Merge clusters and original dataset
        h_clusters_label = pd.merge(h_clusters, df_enrich, on=index_var)
        h_clusters_label = h_clusters_label.rename(columns={"main form": "lemma"})
        h_clusters_names = hierarchical_clusters_label(h_clusters_label)

        # Make treemap
        treemap = topics_treemap(nested_topics=h_clusters_names, index_var=index_var)

        # Make sunburst
        sunburst = topics_sunburst(nested_topics=h_clusters_names, index_var=index_var)

        # Make Sankey Diagram
        sankey = make_sankey(h_clusters_names, field="Dataset", index_var=index_var)

        # Save the arguments in the self
        self.df = df
        self.text_var = text_var
        self.index_var = index_var
        self.sample_size = sample_size
        self.sample_terms = sample_terms
        self.embeddings_model = embeddings_model
        self.ngrams = ngrams
        self.ents = ents
        self.language = language

        self.terms = terms
        self.h_clusters_number = h_clusters
        self.h_clusters_names = h_clusters_names
        self.embeddings_raw = embeddings_reduced
        self.embeddings = df_emb.drop("level_0", axis=1)

        self.treemap = treemap
        self.sunburst = sunburst
        self.sankey = sankey
        self.indexed_terms = df_indexed_full

        return self

    def visualize_embeddings(
        self, nested_level: int = 0, width: int = 1000, height: int = 1000
    ):

        """Visualize the embeddings in 2D based on nested level.
        There is an hover for the text and clusters have names.

        """

        res = pd.merge(
            self.embeddings,
            self.h_clusters_names,
            on=self.index_var,
        )
        res = pd.merge(res, self.df, on=self.index_var)

        if not hasattr(self, "embeddings_2d"):
            self.embeddings_2d = umap.UMAP(n_components=2).fit_transform(
                res[[0, 1, 2, 3, 4]]
            )

        res["dim_1"] = self.embeddings_2d[:, 0]
        res["dim_2"] = self.embeddings_2d[:, 1]

        res[f"level_{nested_level}"] = res[f"level_{nested_level}"].astype(object)
        res[self.text_var] = res[self.text_var].apply(lambda x: wrap_by_word(x, 10))

        res["clusters"] = (
            res[f"level_{nested_level}"].astype(str)
            + " - "
            + res[f"lemma_{nested_level}"]
        )
        res = res.dropna()

        fig = px.scatter(
            res,
            x="dim_1",
            y="dim_2",
            color="clusters",
            hover_data=[self.text_var],
            width=width,
            height=height,
        )

        return fig

    def get_centroid_documents(
        self, nested_level: int = 0, top_elements: int = 2
    ) -> pd.DataFrame:
        """For a given nestednness, Get the centroid element(s), the farest element
        and the cluster radius (farest element - centroid element)


        Returns
        -------
        pd.DataFrame
            the centroid_docs are separeated by ' || '

        """

        # Merge embeddings and text_var
        df_centroid = pd.merge(self.embeddings, self.df, on=self.index_var)

        # merge embeddings & clusters names
        df_centroid = pd.merge(df_centroid, self.h_clusters_names, on=self.index_var)
        df_centroid = df_centroid.rename(
            columns={0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
        )

        res = find_centroids(
            df_centroid,
            text_var=self.text_var,
            cluster_var=f"level_{nested_level}",
            top_elements=top_elements,
            dim_lenght=5,
        )

        return res
