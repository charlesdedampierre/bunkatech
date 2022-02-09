import pandas as pd
import matplotlib.pyplot as plt
from dirty_cat import SimilarityEncoder
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list

pd.options.mode.chained_assignment = None


def fuzzy_matching(
    features: list,
    model: str = "dirty_cat",
    thresh: int = 2,
    output_figure: bool = True,
    destination_path: str = "",
):

    """
    This function takes a Dataframe column with semantic values as an input and sort the values based on their semantic
    similarities.

    Args:
            features (list): list of features to analyze
            model (str): name of the model to use for embedding (bert or SimilarityEncoder)
            thresh (int, optional): Threshold for clusterization. Defaults to 2.
            output_figure (bool, optional): Ouput a figure or not. Defaults to False.

    Raises:
            ValueError: if the model does not exist

    Returns:
            pd.DataFrame: a DataFrame with the following columns:
                    - label_cluster
                    - clustcol
                    - features

    """
    data = pd.DataFrame(features, columns=["features"])
    data["features"] = data["features"].apply(lambda x: str(x).lower())

    freq = data["features"].value_counts().rename("freq").reset_index()
    freq.columns = ["label", "freq"]

    # drop duplicates and nan
    data["features"] = data["features"].drop_duplicates()
    feature_list = data["features"].dropna()
    sorted_values = feature_list.sort_values().unique()

    if model == "sbert":
        emb_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        embeddings = emb_model.encode(list(sorted_values), show_progress_bar=True)
        list_label = list(sorted_values)

    elif model == "dirty_cat":  # Semantic similarities using SimilarityEncoder
        emb_model = SimilarityEncoder(similarity="ngram")
        embeddings = emb_model.fit_transform(sorted_values.reshape(-1, 1))
        list_label = list(emb_model.categories_[0])

    else:
        raise ValueError("Chose a right model name")

    # Hierarchic clustering method
    Z = linkage(embeddings, "ward")

    # Create a figure with the clusters
    if output_figure == True:
        if len(sorted_values) > 500:
            fig = plt.figure(figsize=(100, 10))
        else:
            fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z, labels=list_label)
        plt.axhline(y=thresh, c="grey", lw=1, linestyle="dashed")
        plt.savefig(
            destination_path + f"/hierachical_clustering_{model}.jpeg",
            format="jpeg",
            bbox_inches="tight",
            dpi=300,
        )

    # Get the list of sorted list of words
    list_index = leaves_list(Z)
    new = pd.DataFrame(sorted_values).reindex(list_index)
    new = new.reset_index()
    new.columns = ["idx", "label"]

    # Determine a threshold for clusterization
    assigncol = pd.DataFrame(list(fcluster(Z, thresh, "distance")))
    assigncol["idx"] = assigncol.index
    assigncol.columns = ["clustcol", "idx"]
    assigncol = assigncol.reindex(list_index)

    # Merge the dataframe and output the final excel file
    fin = pd.merge(new, assigncol, on="idx")
    fin = pd.merge(fin, freq, on=["label"])
    fin = fin.sort_values(["clustcol", "freq"], ascending=(True, False))

    if model == "dirty_cat":
        cluster_label = fin.groupby("clustcol").head(1)
    else:
        cluster_label = (
            fin.groupby("clustcol")["label"]
            .apply(lambda x: " | ".join(x[:3]))
            .reset_index()
        )
    cluster_label = cluster_label[["label", "clustcol"]]
    cluster_label.columns = ["label_cluster", "clustcol"]

    # Merge with cluster label
    fin = pd.merge(fin, cluster_label, on="clustcol")
    fin = fin.drop("idx", axis=1)

    return fin
