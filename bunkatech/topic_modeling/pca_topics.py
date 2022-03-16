import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# Make a class
# project the documents
# Better terms extract


def get_top_terms_pca(full_concat, pca, head_tail_number=5) -> pd.DataFrame:

    """

    Get the top and tail terms that are closest to every axis in the PCA
    full_concat is the concat of pca_dimensions and terms dimensions based on
    Tf-Idf


    """

    # Get the cosinus similarity between elements
    full_concat_t = full_concat.T  # Transpose to get the columns as index
    cos = cosine_similarity(full_concat_t)

    """ 
    # Scale between -1 and 1 for more explainability
    scaler = MinMaxScaler(feature_range=(-1, 1))
    cos = scaler.fit_transform(cos)
    """
    df_cos = pd.DataFrame(cos)

    df_cos.columns = full_concat.columns
    df_cos.index = full_concat.columns
    df_cos = df_cos.iloc[
        : pca.n_components_, pca.n_components_ :
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
    pca_var = pd.DataFrame(pca.explained_variance_, columns=["explained_variance"])
    pca_var["dimensions"] = [f"pca_{x}" for x in range(pca.n_components_)]
    fin_cos = pd.merge(fin_cos, pca_var, on="dimensions")

    return fin_cos


def pca_topics(data, text_var, n_components=4, head_tail_number=5, language="english"):
    """
    For every PCA axix, get the top and tail terms to describe them
    """

    data = data[data[text_var].notna()]

    X = data[text_var].to_list()
    model_tfidf = TfidfVectorizer(max_features=10000, stop_words=language)
    X_tfidf = model_tfidf.fit_transform(X)

    X = np.array(X_tfidf.todense())
    X = X - X.mean(axis=0)

    # Get the PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)
    variables_names = model_tfidf.get_feature_names()
    pca_columns = [f"pca_{x}" for x in range(n_components)]

    # concat the X_pca and the X_terms
    df_X_pca = pd.DataFrame(X_pca, columns=pca_columns)
    df_X = pd.DataFrame(X, columns=variables_names)
    full_concat = pd.concat([df_X_pca, df_X], axis=1)

    res = get_top_terms_pca(full_concat, pca, head_tail_number=head_tail_number)

    return res


if __name__ == "__main__":

    data = pd.read_csv(
        "/Users/charlesdedampierre/Desktop/ENS Projects/imaginary-world/db_film_iw (2).csv",
        index_col=[0],
    )
    data = data.sample(5000)

    # Get with the terms dimensions PCA are computed on

    res = pca_topics(
        data,
        text_var="description",
        n_components=10,
        head_tail_number=10,
        language="english",
    )

    print(res)
