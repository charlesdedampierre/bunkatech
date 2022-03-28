import pandas as pd
import numpy as np

import plotly
import plotly.express as px
import warnings
import umap

from .semantics.extract_terms_old import extract_terms_df
from .semantics.indexer import indexer
from .semantics.get_embeddings import get_embeddings
from .semantics.semantic_folding import semantic_folding


from .hierarchical_clusters import hierarchical_clusters
from .hierarchical_clusters_label import hierarchical_clusters_label

from .visualization.make_bubble import wrap_by_word
from .visualization.sankey import make_sankey
from .visualization.topics_treemap import topics_treemap
from .visualization.topics_sunburst import topics_sunburst
from .visualization.topics_nested import topics_nested


from .networks.centroids import find_centroids
from .search.fts5_search import fts5_search


warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


class NestedTopicModeling:
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
        folding=None,
        ngrams=(1, 2),
        ents=False,
        ncs=False,
        language="en",
        db_path=".",
    ):

        df = df.sample(min(len(df), sample_size))
        df = df[df[text_var].notna()]  # cautious not to let any nan
        df = df.reset_index(drop=True)

        if folding is not None:

            embeddings_reduced = semantic_folding(
                folding,
                df,
                text_var=text_var,
                model=embeddings_model,
                dimension_folding=5,
                folding_only=True,
            )
        else:
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
            ncs=ncs,  # nouns
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

        self.indexed_terms = df_indexed_full

        return self

    def make_nested_maps(
        self,
        size_rule="equal_size",
        map_type="treemap",
        width=1000,
        height=1000,
        query=None,
    ):
        """Create Maps to display information

        Parameters
        ----------
        size_rule : str, optional
            _description_, by default "equal_size"
        map_type : str, optional
            _description_, by default "treemap"
        width : int, optional
            _description_, by default 1000
        height : int, optional
            _description_, by default 1000
        query : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """

        if query is not None:
            docs = self.df[self.text_var].to_list()
            # Make the term search among the documents
            res_search = fts5_search(query, docs)

            # Merge the results with the text_var
            df_cluster_names_filter = pd.merge(
                self.df, res_search, left_on=self.text_var, right_on="docs"
            )

            # Merge the ids with the nested clusters if information
            df_cluster_names_filter = pd.merge(
                self.h_clusters_names,
                df_cluster_names_filter[[self.index_var]],
                on=self.index_var,
            )

            # group by the more nested topics
            count_search = (
                df_cluster_names_filter.groupby(["lemma_2"])
                .agg(score_norm=(self.index_var, "count"))
                .reset_index()
            )
        else:
            count_search = None

        if map_type == "treemap":
            # Make treemap
            map = topics_nested(
                nested_topics=self.h_clusters_names,
                index_var=self.index_var,
                size_rule=size_rule,
                width=width,
                height=height,
                count_search=count_search,
                map_type="treemap",
            )

        elif map_type == "sunburst":
            # Make sunburst
            map = topics_nested(
                nested_topics=self.h_clusters_names,
                index_var=self.index_var,
                size_rule=size_rule,
                width=width,
                height=height,
                count_search=count_search,
                map_type="sunburst",
            )

        elif map_type == "sankey":
            # Make Sankey Diagram
            map = make_sankey(
                self.h_clusters_names, field="Dataset", index_var=self.index_var
            )

        else:
            raise ValueError(
                f' "{map_type}" is not a correct value. Please enter a correct map_type value such as "Treemap", "Sunburst" or "Sankey"'
            )

        return map

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
