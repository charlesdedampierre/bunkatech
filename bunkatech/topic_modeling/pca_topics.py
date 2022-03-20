from re import X
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import plotly.express as px
import warnings
from ..basic_class import BasicSemantics

pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)


def wrap_by_word(string, n_words):
    """returns a string where \\n is inserted between every n words"""
    try:
        a = string.split()
        ret = ""
        for i in range(0, len(a), n_words):
            ret += " ".join(a[i : i + n_words]) + "<br>"
    except:
        pass

    return ret


# Make a class
# project the documents
# Better terms extract


class PCATopic(BasicSemantics):
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
        terms_ngrams=(1, 2),
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

    def fit(self, n_components=4, head_tail_number=5):
        self.pca_computation(n_components=n_components)
        self.pca_topics = self.get_top_terms_pca(head_tail_number=head_tail_number)

        return self.pca_topics

    def pca_computation(self, n_components=4):
        """
        For every PCA axix, get the top and tail terms to describe them
        """

        # Get the cosine similarity between all embedded terms
        df_emb = cosine_similarity(self.terms_embeddings)
        df_emb = pd.DataFrame(df_emb)
        df_emb.columns = self.terms_embeddings.index
        df_emb.index = self.terms_embeddings.index

        # Merge with index
        df_emb = df_emb.reset_index()
        df_emb = df_emb.rename(columns={"index": "main form"})

        # Merge with the indexed terms. Documents that have no index terms get the mean of
        # the dataset
        res = pd.merge(
            df_emb, self.df_indexed[["main form"]].reset_index(), on="main form"
        ).drop("main form", axis=1)

        # merge with the original data to get the index that have no indexed terms.
        # Fill the nan values with the mean of the dataset
        # res = pd.merge(res, self.data[[self.index_var]], on=self.index_var, how="right")
        # res = res.fillna(res.mean())

        # every documents vector is the mean of its components' vectors
        X = res.groupby(self.index_var).mean()

        """
        for col in X.columns:
            X[col] = X[col].fillna(X[col].mean())
        # X = X.fillna(X.mean())
        self.X = X"""

        """ X = self.data[self.text_var].to_list()
        model_tfidf = TfidfVectorizer(max_features=10000, stop_words=language)
        X_tfidf = model_tfidf.fit_transform(X)

        X = np.array(X_tfidf.todense())
        X = X - X.mean(axis=0)"""

        # recompute this part

        # Get the PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X)
        X_pca = self.pca.transform(X)
        # variables_names = model_tfidf.get_feature_names()
        variables_names = self.terms_embeddings.index
        pca_columns = [f"pca_{x}" for x in range(self.pca.n_components_)]

        # concat the X_pca and the X_terms
        self.df_X_pca = pd.DataFrame(X_pca, columns=pca_columns)
        self.df_X_pca.index = X.index

        self.df_X = pd.DataFrame(X, columns=variables_names)
        self.full_concat = pd.concat([self.df_X_pca, self.df_X], axis=1)

        return self.full_concat

    def get_top_terms_pca(self, head_tail_number=5) -> pd.DataFrame:

        """

        Get the top and tail terms that are closest to every axis in the PCA
        full_concat is the concat of pca_dimensions and terms dimensions based on
        Tf-Idf


        """

        # Get the cosinus similarity between elements
        full_concat_t = self.full_concat.T  # Transpose to get the columns as index
        cos = cosine_similarity(full_concat_t)

        """ 
        # Scale between -1 and 1 for more explainability
        scaler = MinMaxScaler(feature_range=(-1, 1))
        cos = scaler.fit_transform(cos)
        """
        df_cos = pd.DataFrame(cos)

        df_cos.columns = self.full_concat.columns
        df_cos.index = self.full_concat.columns
        df_cos = df_cos.iloc[
            : self.pca.n_components_, self.pca.n_components_ :
        ]  # Keep the first dimentsions in rows and get rid of them in column

        df_cos = df_cos.reset_index()
        df_cos = df_cos.melt("index")
        df_cos.columns = ["dimensions", "terms", "cos"]

        df_cos = df_cos.sort_values(
            ["dimensions", "cos"], ascending=(True, False)
        ).reset_index(drop=True)

        # Get the head and tails of cosinus to get the dimensions range
        top_cos = df_cos.groupby("dimensions").head(head_tail_number)
        top_cos["axis"] = "head"

        tail_cos = df_cos.groupby("dimensions").tail(head_tail_number)
        tail_cos["axis"] = "tail"
        tail_cos = tail_cos.sort_values(["dimensions", "cos"])

        fin_cos = pd.concat([top_cos, tail_cos]).reset_index(drop=True)
        fin_cos = fin_cos.sort_values(
            ["dimensions", "cos"], ascending=(True, False)
        ).reset_index(drop=True)

        fin_cos = (
            fin_cos.groupby(["dimensions", "axis"])["terms"]
            .apply(lambda x: " | ".join(x))
            .reset_index()
        )

        # Get the explained variance
        pca_var = pd.DataFrame(
            self.pca.explained_variance_ratio_, columns=["explained_variance_ratio"]
        )
        pca_var["dimensions"] = [f"pca_{x}" for x in range(self.pca.n_components_)]
        fin_cos = pd.merge(fin_cos, pca_var, on="dimensions")

        return fin_cos

    def visualize(self, dim_1, dim_2, width=1000, height=1000):
        # visualize the embeddings

        res = self.df_X_pca[[dim_1, dim_2]].reset_index()
        res = pd.merge(res, self.data, on=self.index_var)
        # res[self.text_var] = self.data[self.text_var].tolist()
        res[self.text_var] = res[self.text_var].apply(lambda x: wrap_by_word(x, 10))

        fig = px.scatter(
            res,
            x=dim_1,
            y=dim_2,
            hover_data=[self.text_var],
            width=width,
            height=height,
        )

        df_axis = self.pca_topics.copy()
        df_axis = df_axis.sort_values(["dimensions", "axis"], ascending=(True, False))

        df_axis = (
            df_axis.groupby("dimensions")["terms"]
            .apply(lambda x: " /// ".join(x))
            .reset_index()
        )

        dim_1_axis = df_axis[df_axis["dimensions"] == dim_1]["terms"].iloc[0]
        dim_2_axis = df_axis[df_axis["dimensions"] == dim_2]["terms"].iloc[0]

        fig.update_layout(
            title="PCA Projection",
            height=height,
            width=width,
            xaxis_title="<--- " + dim_1_axis + " --->",
            yaxis_title="<--- " + dim_2_axis + " --->",
        )

        return fig


if __name__ == "__main__":

    data = pd.read_csv(
        "/Users/charlesdedampierre/Desktop/ENS Projects/imaginary-world/db_film_iw (2).csv",
        index_col=[0],
    )
    data = data.sample(5000)
