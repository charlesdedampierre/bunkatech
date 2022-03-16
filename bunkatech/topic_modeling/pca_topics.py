from re import X
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import plotly.express as px
import warnings

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


class PCATopic:
    def __init__(self, data, text_var) -> None:

        self.data = data
        self.text_var = text_var
        self.data = self.data[self.data[self.text_var].notna()].reset_index(drop=True)

    def fit(self):
        pass

    def pca_computation(self, n_components=4, language="english"):
        """
        For every PCA axix, get the top and tail terms to describe them
        """

        X = self.data[self.text_var].to_list()
        model_tfidf = TfidfVectorizer(max_features=10000, stop_words=language)
        X_tfidf = model_tfidf.fit_transform(X)

        X = np.array(X_tfidf.todense())
        X = X - X.mean(axis=0)

        # recompute this part

        # Get the PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X)
        X_pca = self.pca.transform(X)
        variables_names = model_tfidf.get_feature_names()
        pca_columns = [f"pca_{x}" for x in range(self.pca.n_components_)]

        # concat the X_pca and the X_terms
        self.df_X_pca = pd.DataFrame(X_pca, columns=pca_columns)
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
            self.pca.explained_variance_, columns=["explained_variance"]
        )
        pca_var["dimensions"] = [f"pca_{x}" for x in range(self.pca.n_components_)]
        fin_cos = pd.merge(fin_cos, pca_var, on="dimensions")

        return fin_cos

    def get_topics(self, n_components=4, head_tail_number=5, language="english"):
        self.pca_computation(
            n_components=n_components,
            language=language,
        )
        self.pca_topics = self.get_top_terms_pca(head_tail_number=head_tail_number)

        return self.pca_topics

    def visualize(self, dim_1, dim_2, width=1000, height=1000):
        # visualize the embeddings

        res = self.df_X_pca[[dim_1, dim_2]]
        res[self.text_var] = self.data[self.text_var]
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

    # Get with the terms dimensions PCA are computed on
    pca_topic = PCATopic(data=data, text_var="description")
    res = pca_topic.get_topics(n_components=10, head_tail_number=3, language="english")
    fig = pca_topic.visualize(dim_1="pca_1", dim_2="pca_2", width=1000, height=1000)
    fig.show()
