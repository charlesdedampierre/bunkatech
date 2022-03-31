import pandas as pd
from ..specificity import specificity


def hierarchical_clusters_label(
    final: pd.DataFrame, top_n: int = 3, levels: list = [0, 1, 2]
):
    """This function takes a nested clusters and the indexed terms and compute for every cluster
    the most specific terms.

    Args:
        final (pd.DataFrame): [description]
        top_n (int, optional): top terms. Defaults to 3.
        levels (list, optional): the level of nestedness. Defaults to [0, 1, 2].

    Returns:
        [type]: a dataframe with the the name of every nested cluster
    """

    _, _, toop_1 = specificity(final, f"level_{levels[0]}", "lemma", None, top_n=top_n)
    toop_1 = toop_1.rename(columns={"lemma": f"lemma_{levels[0]}"})
    toop_1 = (
        toop_1.groupby([f"level_{levels[0]}"])[f"lemma_{levels[0]}"]
        .apply(lambda x: " | ".join(x))
        .reset_index()
    )
    new = pd.merge(final, toop_1, on="level_0")
    new = new.drop("lemma", axis=1).drop_duplicates()

    for tr in levels[1:]:
        full_toop = pd.DataFrame()
        for x in set(final[f"level_{tr-1}"]):
            _, _, toop = specificity(
                final[final[f"level_{tr-1}"] == x],
                f"level_{tr}",
                "lemma",
                None,
                top_n=top_n,
            )
            toop = toop.rename(columns={"lemma": f"lemma_{tr}"})
            toop = (
                toop.groupby([f"level_{tr}"])[f"lemma_{tr}"]
                .apply(lambda x: " | ".join(x))
                .reset_index()
            )
            full_toop = full_toop.append(toop)
        new = pd.merge(new, full_toop, on=f"level_{tr}", how="left")
    return new


# get rid of prevous clusters
