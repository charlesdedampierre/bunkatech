import pandas as pd
import numpy as np
import plotly.graph_objs as go
import random

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from ..basic_class import BasicSemantics
from sentence_transformers import SentenceTransformer

pd.options.mode.chained_assignment = None


def wrap_by_word(string, n_words):
    """returns a string where \\n is inserted between every n words"""
    a = string.split()
    ret = ""
    for i in range(0, len(a), n_words):
        ret += " ".join(a[i : i + n_words]) + "<br>"

    return ret


class Origami(BasicSemantics):
    """This class creates a two dimentional space where it projects terms or documents.
    The two spaces are created from words continuum. such as 'good-bad' and 'man-woman'.
    We then analyse whether these is a correlation or not in the space computes by transformers
    (like Sbert)
    """

    def __init__(
        self,
        data,
        text_var,
        index_var,
        extract_terms=True,
        terms_embedding=True,
        docs_embedding=True,
        sample_size_terms=500,
        terms_limit=500,
        terms_ents=True,
        terms_ngrams=(2, 2),
        terms_ncs=True,
        terms_include_pos=["NOUN", "PROPN", "ADJ"],
        terms_include_types=["PERSON", "ORG"],
        terms_embedding_model="distiluse-base-multilingual-cased-v1",
        docs_embedding_model="tfidf",
        language="en",
        terms_path=None,
        terms_embeddings_path=None,
        docs_embeddings_path=None,
    ) -> None:

        BasicSemantics.__init__(
            self,
            data=data,
            text_var=text_var,
            index_var=index_var,
            terms_path=terms_path,
            terms_embeddings_path=terms_embeddings_path,
            docs_embeddings_path=docs_embeddings_path,
        )

        BasicSemantics.fit(
            self,
            extract_terms=extract_terms,
            terms_embedding=terms_embedding,
            docs_embedding=docs_embedding,
            sample_size_terms=sample_size_terms,
            terms_limit=terms_limit,
            terms_ents=terms_ents,
            terms_ngrams=terms_ngrams,
            terms_ncs=terms_ncs,
            terms_include_pos=terms_include_pos,
            terms_include_types=terms_include_types,
            terms_embedding_model=terms_embedding_model,
            docs_embedding_model=docs_embedding_model,
            language=language,
        )

    def compute_projection_embeddings(
        self, projection_1=["cooperative", "salaud"], projection_2=["woman", "man"]
    ):
        """Compute embeddings of the newly projected terms and
        associate them to the initial embeddings matrix

        """

        projection_all = projection_1 + projection_2
        self.projection_all = projection_all

        # index the new projected terms
        self.index_terms(projection=True)

        # Encode the results
        model = SentenceTransformer(self.terms_embedding_model)
        projection_embeddings = model.encode(projection_all, show_progress_bar=True)

        full_terms_embeddings = np.concatenate(
            [self.terms_embeddings, projection_embeddings]
        )
        full_terms = self.terms["main form"].to_list() + projection_all
        full_terms_prepro = [x.lower() for x in full_terms]

        df_bert = cosine_similarity(full_terms_embeddings)
        df_bert = pd.DataFrame(df_bert)
        df_bert.columns = full_terms_prepro
        df_bert.index = full_terms_prepro

        # Duplicates because fome terms may have already been embedded
        df_bert = df_bert.loc[:, ~df_bert.columns.duplicated()]

        return df_bert

    def origami_projection(
        self,
        projection_1: list,
        projection_2: list,
        height: int = 1000,
        width: int = 1000,
        regression=True,
        type="terms",
    ):
        """

        Create the projection Space with plotly based on two queries: projection_1 & projection_2

        """

        projection_str_1 = "-".join(projection_1)
        projection_str_2 = "-".join(projection_2)

        self.df_bert = self.compute_projection_embeddings(projection_1, projection_2)

        # Select the dimentions of interetst in all the similarity matric
        df_proj = self.df_bert[projection_1 + projection_2]

        df_proj[projection_str_1] = df_proj[projection_1[0]] - df_proj[projection_1[1]]
        df_proj[projection_str_2] = df_proj[projection_2[0]] - df_proj[projection_2[1]]

        # Scale the results from -1 to 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_proj["term"] = df_proj.index

        df_proj[projection_str_1] = scaler.fit_transform(
            df_proj[projection_str_1].values.reshape(-1, 1)
        )
        df_proj[projection_str_2] = scaler.fit_transform(
            df_proj[projection_str_2].values.reshape(-1, 1)
        )

        if type == "documents":
            # Merge with the original indexed data. The term is the key here
            fin = pd.merge(
                df_proj[["term", projection_str_1, projection_str_2]],
                self.df_indexed[["main form"]].reset_index(),
                left_on="term",
                right_on="main form",
            )

            # Compute the mean for every id: every id is the mean of all the texts it contains
            res = (
                fin.groupby([self.index_var])
                .agg(
                    projection_str=(projection_str_1, "mean"),
                    projection_str_2=(projection_str_2, "mean"),
                )
                .reset_index()
            )
            res = res.rename(
                columns={
                    "projection_str": projection_str_1,
                    "projection_str_2": projection_str_2,
                }
            )

            # Get the text var
            res = pd.merge(
                res, self.data[[self.index_var, self.text_var]], on=self.index_var
            )

            # Rescale
            res[projection_str_1] = scaler.fit_transform(res[[projection_str_1]])
            res[projection_str_2] = scaler.fit_transform(res[[projection_str_2]])

            final_proj = res.copy()
            name_var = self.text_var

            # Get rid of nan values
            final_proj = final_proj[final_proj[name_var].notna()]
            final_proj[name_var] = final_proj[name_var].apply(
                lambda x: wrap_by_word(x, 10)
            )

        elif type == "terms":
            final_proj = df_proj.copy()
            name_var = "term"

        else:
            raise ValueError("Please chose between 'terms' or 'documents'")

        # Plot everything
        fig = go.Figure()

        # Plot two axes x and y to visualisy scale the results (like PCA projection)
        trace_1 = go.Scatter(
            x=[-1.1, 1.1],
            y=[0, 0],
            mode="lines",
            line_color="grey",
            name=projection_str_1,
        )

        trace_2 = go.Scatter(
            x=[0, 0],
            y=[-1.1, 1.1],
            mode="lines",
            line_color="grey",
            name=projection_str_2,
        )

        # Plot the elements
        trace_scatter = go.Scatter(
            x=final_proj[projection_str_1],
            y=final_proj[projection_str_2],
            text=final_proj[name_var],
            mode="markers",
            name=name_var,
        )

        fig.add_trace(trace_scatter)
        fig.add_trace(trace_1)
        fig.add_trace(trace_2)

        if regression is True:
            reg = LinearRegression().fit(
                np.vstack(final_proj[projection_str_1]), final_proj[projection_str_2]
            )
            final_proj["bestfit"] = reg.predict(np.vstack(final_proj[projection_str_1]))

            fig.add_trace(
                go.Scatter(
                    name="linear Regression",
                    x=final_proj[projection_str_1],
                    y=final_proj["bestfit"],
                    mode="lines",
                )
            )

        fig.update_layout(
            title="Semantic Origami",
            height=height,
            width=width,
            xaxis_title="<--- " + " | ".join(reversed(projection_1)) + " --->",
            yaxis_title="<--- " + " | ".join(reversed(projection_2)) + " --->",
        )

        self.df_fig = final_proj

        return fig

    def origami_projection_unique(
        self,
        projection_1: list,
        height: int = 1000,
        width: int = 1000,
        type="terms",
        dispersion=True,
        barometer=True,
    ):
        """

        Create the projection Space with plotly based on two queries: projection_1 & projection_2

        """

        projection_str_1 = "-".join(projection_1)
        self.df_bert = self.compute_projection_embeddings(projection_1)

        # Select the dimentions of interetst in all the similarity matric
        df_proj = self.df_bert[projection_1]
        df_proj[projection_str_1] = df_proj[projection_1[0]] - df_proj[projection_1[1]]

        # Scale the results from -1 to 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_proj["term"] = df_proj.index
        df_proj[projection_str_1] = scaler.fit_transform(
            df_proj[projection_str_1].values.reshape(-1, 1)
        )

        df_proj["project_angle"] = np.sqrt(
            1 - df_proj[projection_str_1] ** 2
        )  # Pythagore

        scaler_pyth = MinMaxScaler(feature_range=(0, 1))
        df_proj["project_angle"] = scaler_pyth.fit_transform(
            df_proj["project_angle"].values.reshape(-1, 1)
        )

        if type == "documents":
            # group by term and mean of the vectors
            # Merge with the original indexed data. The term is the key here
            fin = pd.merge(
                df_proj[["term", projection_str_1]],
                self.df_indexed[["main form"]].reset_index(),
                left_on="term",
                right_on="main form",
            )

            # Compute the mean for every id: every id is the mean of all the texts it contains
            res = (
                fin.groupby([self.index_var])
                .agg(projection_str=(projection_str_1, "mean"))
                .reset_index()
            )
            res = res.rename(
                columns={
                    "projection_str": projection_str_1,
                }
            )

            # Get the text_var by merging with the original data
            res = pd.merge(
                res, self.data[[self.index_var, self.text_var]], on=self.index_var
            )

            res["project_angle"] = np.sqrt(1 - res[projection_str_1] ** 2)
            res["project_angle"] = scaler_pyth.fit_transform(res[["project_angle"]])

            # Rescale
            res[projection_str_1] = scaler.fit_transform(res[[projection_str_1]])

            final_proj = res.copy()
            name_var = self.text_var

            # Get rid of nan values
            final_proj = final_proj[final_proj[name_var].notna()]
            final_proj[name_var] = final_proj[name_var].apply(
                lambda x: wrap_by_word(x, 10)
            )

        elif type == "terms":
            final_proj = df_proj.copy()
            name_var = "term"

        else:
            raise ValueError("Please chose between 'terms' or 'documents'")
        if dispersion:
            # Random the data to make them more visible
            final_proj["project_angle"] = final_proj["project_angle"].apply(
                lambda x: x
                * random.uniform(
                    0.001,
                    1,
                )
            )

        # Plot everything
        fig = go.Figure()

        # Plot two axes x and y to visualisy scale the results (like PCA projection)
        trace_1 = go.Scatter(
            x=[-1.1, 1.1],
            y=[0, 0],
            mode="lines",
            line_color="grey",
            name=projection_str_1,
        )

        trace_2 = go.Scatter(
            x=[0, 0],
            y=[0, 1.1],
            mode="lines",
            line_color="grey",
            name="project_angle",
        )

        # Plot the elements
        trace_scatter = go.Scatter(
            x=final_proj[projection_str_1],
            y=final_proj["project_angle"],
            text=final_proj[name_var],
            mode="markers",
            name=name_var,
        )

        fig.add_trace(trace_scatter)
        fig.add_trace(trace_1)
        fig.add_trace(trace_2)

        if barometer:
            # Get the barometer Line
            baro_mean = final_proj[projection_str_1].mean()
            baro_mean_angle = np.sqrt(1 - baro_mean ** 2)
            trace_barometer = go.Scatter(
                x=[0.0, baro_mean, None],
                y=[0.0, baro_mean_angle, None],
                mode="lines",
                line_width=3,
                line_color="crimson",
                name="barometer",
            )
            fig.add_trace(trace_barometer)

        fig.update_layout(
            title="Semantic Origami",
            height=height,
            width=width,
            xaxis_title="<--- " + " | ".join(reversed(projection_1)) + " --->",
        )

        self.df_fig = final_proj

        return fig
