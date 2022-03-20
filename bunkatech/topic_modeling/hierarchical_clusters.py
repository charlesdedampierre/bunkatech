from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def hierarchical_clusters(df: pd.DataFrame) -> pd.DataFrame:

    """Make nested clusters. 8 level for the first layer, 3 level for the second layer
    and 2 levels for the last layer

    Args:
        df (pd.DataFrame): Dataframe with the index_var, the embeddings (5 dimensions)

    Returns:
        (pd.DataFrame):  with three columns (level_0, level_1, level_2) for every layer
    """
    # level 0

    try:
        emb_col = list(np.arange(0, 5))
        level_0 = (
            KMeans(n_clusters=8, random_state=0).fit(df[emb_col]).labels_
        )  # first layer
        df["level_0"] = level_0
        df = df.sort_values("level_0")
    except:
        emb_col = list(np.arange(0, 5))
        emb_col = [str(x) for x in emb_col]
        level_0 = (
            KMeans(n_clusters=8, random_state=0).fit(df[emb_col]).labels_
        )  # first layer
        df["level_0"] = level_0
        df = df.sort_values("level_0")

    # level 1
    levels_1 = []
    x = 0
    for cluster in set(df["level_0"]):
        new_df = df[df[f"level_0"] == cluster][emb_col]
        level_1 = (
            KMeans(n_clusters=4, random_state=0).fit(new_df).labels_
        )  # second layer
        if x > 0:
            dict = {0: x, 1: x + 1, 2: x + 2, 3: x + 3}

            level_1 = [dict[item] for item in level_1]
            levels_1.append(level_1)
        else:
            levels_1.append(level_1)
        x += 4

    flat_level_1 = [item for sublist in levels_1 for item in sublist]

    df["level_1"] = flat_level_1
    df = df.sort_values("level_1")

    # level 2
    levels_2 = []
    x = 0
    for cluster in set(df["level_1"]):
        new_df = df[df[f"level_1"] == cluster][emb_col]
        try:
            level_2 = (
                KMeans(n_clusters=2, random_state=0).fit(new_df).labels_
            )  # 3rd layer
        except:
            level_2 = [x for x in range(len(new_df))]
        if x > 0:
            dict = {0: x, 1: x + 1}

            level_2 = [dict[item] for item in level_2]
            levels_2.append(level_2)
        else:
            levels_2.append(level_2)
        x += 2

    flat_level_2 = [item for sublist in levels_2 for item in sublist]

    df["level_2"] = flat_level_2

    df = df.sort_values(["level_0", "level_1", "level_2"])
    # drop the vectors
    df = df.drop(emb_col, axis=1)
    df = df.reset_index(drop=True)
    return df
