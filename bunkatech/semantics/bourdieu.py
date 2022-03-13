import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from .extract_terms import extract_terms_df
from .indexer import indexer
from sentence_transformers import SentenceTransformer

pd.options.mode.chained_assignment = None


def wrap_by_word(string, n_words):
    """returns a string where \\n is inserted between every n words"""
    a = string.split()
    ret = ""
    for i in range(0, len(a), n_words):
        ret += " ".join(a[i : i + n_words]) + "<br>"

    return ret


class Bourdieu:
    def __init__(self) -> None:
        pass

    def fit(self, data, text_var, index_var):
        data = data[data[text_var].notna()]
        self.data = data
        self.text_var = text_var
        self.index_var = index_var
        return self

    def extract_terms(
        self,
        sample_size,
        limit,
        ents=True,
        ncs=True,
        ngrams=(1, 2),
        include_pos=["NOUN", "PROPN", "ADJ"],
        include_types=["PERSON", "ORG"],
        language="en",
        db_path=".",
    ):
        terms = extract_terms_df(
            self.data,
            text_var=self.text_var,
            limit=limit,
            sample_size=sample_size,
            ngs=True,  # ngrams
            ents=ents,  # entities
            ncs=ncs,  # nouns
            drop_emoji=True,
            remove_punctuation=False,
            ngrams=ngrams,
            include_pos=include_pos,
            include_types=include_types,
            language=language,
        )

        terms["main form"] = terms["main form"].apply(lambda x: x.lower())
        self.terms = terms

        # index terms
        df_terms = self.terms.copy()
        df_terms["text"] = df_terms["text"].apply(lambda x: x.split(" | "))
        df_terms = df_terms.explode("text").reset_index(drop=True)
        list_terms = df_terms["text"].tolist()

        # Index the extracted terms
        df_indexed = indexer(
            self.data[self.text_var].tolist(), list_terms, db_path=db_path
        )

        df_indexed_full = pd.merge(
            df_indexed, df_terms, left_on="words", right_on="text"
        )
        df_indexed_full = df_indexed_full[["docs", "lemma", "main form", "text"]].copy()
        df_enrich = pd.merge(
            self.data, df_indexed_full, left_on=self.text_var, right_on="docs"
        )
        self.df_indexed = df_enrich

        return terms

    def sbert_embedding(self, bert_model="distiluse-base-multilingual-cased-v1"):

        self.terms["bindex"] = self.terms.index

        # embed the terms with sbert
        model = SentenceTransformer(bert_model)
        docs = list(self.terms["main form"])
        terms_embeddings = model.encode(docs, show_progress_bar=True)

        # Lower all the terms
        docs_prepro = [x.lower() for x in docs]

        # Compute Similarity between embedded extracted terms
        df_bert = cosine_similarity(terms_embeddings)
        df_bert = pd.DataFrame(df_bert)
        df_bert.columns = docs_prepro
        df_bert.index = docs_prepro

        self.df_bert = df_bert

        return df_bert

    def bourdieu_projection_terms(
        self,
        projection: list,
        projection_2: list,
        height: int = 1000,
        width: int = 1000,
        regression=True,
    ):

        projection_str = "-".join(projection)
        projection_str_2 = "-".join(projection_2)

        # Select the dimentions of interetst in all the similarity matric
        df_proj = self.df_bert[projection + projection_2]
        df_proj[projection_str] = df_proj[projection[0]] - df_proj[projection[1]]
        df_proj[projection_str_2] = df_proj[projection_2[0]] - df_proj[projection_2[1]]

        # Scale the results from -1 to 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_proj[projection_str] = scaler.fit_transform(
            df_proj[projection_str].values.reshape(-1, 1)
        )
        df_proj[projection_str_2] = scaler.fit_transform(
            df_proj[projection_str_2].values.reshape(-1, 1)
        )

        # Visualize the results
        df_proj["term"] = df_proj.index

        fig = go.Figure()

        # Plot two axes x and y to visualisy scale the results (like PCA projection)
        trace_1 = go.Scatter(
            x=[-1.1, 1.1],
            y=[0, 0],
            mode="lines",
            line_color="grey",
            name=projection_str,
        )

        trace_2 = go.Scatter(
            x=[0, 0],
            y=[-1.1, 1.1],
            mode="lines",
            line_color="grey",
            name=projection_str_2,
        )

        trace_scatter = go.Scatter(
            x=df_proj[projection_str],
            y=df_proj[projection_str_2],
            text=df_proj["term"],
            mode="markers",
            name="terms",
        )

        fig.add_trace(trace_scatter)
        fig.add_trace(trace_1)
        fig.add_trace(trace_2)

        if regression is True:
            reg = LinearRegression().fit(
                np.vstack(df_proj[projection_str]), df_proj[projection_str_2]
            )
            df_proj["bestfit"] = reg.predict(np.vstack(df_proj[projection_str]))

            fig.add_trace(
                go.Scatter(
                    name="linear Regression",
                    x=df_proj[projection_str],
                    y=df_proj["bestfit"],
                    mode="lines",
                )
            )

        fig.update_layout(
            title="Bourdieu Projection",
            height=height,
            width=width,
            xaxis_title="<--- " + " | ".join(reversed(projection)) + " --->",
            yaxis_title="<--- " + " | ".join(reversed(projection_2)) + " --->",
        )

        self.df_terms_fig = df_proj

        return fig

    def bourdieu_projection_documents(
        self,
        projection,
        projection_2,
        width=1000,
        height=1000,
        regression=True,
    ):

        projection_str = "-".join(projection)
        projection_str_2 = "-".join(projection_2)

        # Only select terms of interests in the similarity matrix
        df_proj = self.df_bert[projection + projection_2]
        df_proj[projection_str] = df_proj[projection[0]] - df_proj[projection[1]]
        df_proj[projection_str_2] = df_proj[projection_2[0]] - df_proj[projection_2[1]]

        # Rescale the data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_proj["term"] = df_proj.index
        df_proj = df_proj.sort_values(projection_str_2, ascending=True)
        df_proj[projection_str] = scaler.fit_transform(
            df_proj[projection_str].values.reshape(-1, 1)
        )
        df_proj[projection_str_2] = scaler.fit_transform(
            df_proj[projection_str_2].values.reshape(-1, 1)
        )
        # Merge with the original data
        fin = pd.merge(
            df_proj[["term", projection_str, projection_str_2]],
            self.df_indexed[[self.index_var, "main form", self.text_var]],
            left_on="term",
            right_on="main form",
        )
        # Compute the mean for every id: every id is the mean of all the texts it contains
        res = (
            fin.groupby([self.index_var, self.text_var])
            .agg(
                projection_str=(projection_str, "mean"),
                projection_str_2=(projection_str_2, "mean"),
            )
            .reset_index()
        )
        res = res.rename(
            columns={
                "projection_str": projection_str,
                "projection_str_2": projection_str_2,
            }
        )

        res["text_var_prep"] = res[self.text_var].apply(lambda x: wrap_by_word(x, 10))

        # Make Figures
        fig = go.Figure()

        trace_1 = go.Scatter(
            x=[-1.1, 1.1],
            y=[0, 0],
            mode="lines",
            line_color="grey",
            name=projection_str,
        )

        trace_2 = go.Scatter(
            x=[0, 0],
            y=[-1.1, 1.1],
            mode="lines",
            line_color="grey",
            name=projection_str_2,
        )

        trace_scatter = go.Scatter(
            x=res[projection_str],
            y=res[projection_str_2],
            text=res["text_var_prep"],
            mode="markers",
            name=self.text_var,
        )

        fig.add_trace(trace_scatter)
        fig.add_trace(trace_1)
        fig.add_trace(trace_2)

        if regression is True:
            reg = LinearRegression().fit(
                np.vstack(res[projection_str]), res[projection_str_2]
            )
            res["bestfit"] = reg.predict(np.vstack(res[projection_str]))

            fig.add_trace(
                go.Scatter(
                    name="linear Regression",
                    x=res[projection_str],
                    y=res["bestfit"],
                    mode="lines",
                )
            )

        fig.update_layout(
            title="Bourdieu Projection",
            height=height,
            width=width,
            xaxis_title="<--- " + " | ".join(reversed(projection)) + " --->",
            yaxis_title="<--- " + " | ".join(reversed(projection_2)) + " --->",
        )

        self.df_doc_fig = res

        return fig
