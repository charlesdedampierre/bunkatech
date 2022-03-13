import pandas as pd
import numpy as np
import plotly.express as px


def topics_nested(
    nested_topics: pd.DataFrame,
    index_var,
    size_rule="topic_documents",
    max_depth=2,
    width=1000,
    height=1000,
    popularity_var=None,
    count_search=None,
    map_type="treemap",
):
    """Create Treemaps from the df_nested_topics.

    its stucture is as follow:

    id cluster_1 cluster_2 cluster_3/

    Parameters
    ----------
    nested_topics : pd.DataFrame
        _description_
    index_var : _type_
        _description_
    size_rule : str, optional
        The size indicates what should be the rule to create the size of the squares

    topic_documents: count the number of documents inside every topics
    equal_size: give the same number to every topics.
    popularity: normalize the size by the number of likes for instance


        , by default "topic_size"

    Returns
    -------
    _type_
        _description_
    """

    lemma = [f"lemma_{x}" for x in range(max_depth + 1)]

    if size_rule == "topic_documents":
        group = (
            nested_topics.groupby("lemma_2")
            .agg(topic_size=(index_var, "count"))
            .reset_index()
        )

    elif size_rule == "equal_size":
        group = (
            nested_topics.groupby("lemma_2")
            .agg(topic_size=(index_var, "count"))
            .reset_index()
        )
        group["topic_size"] = 1

    elif size_rule == "popularity":
        group = (
            nested_topics.groupby("lemma_2")
            .agg(topic_size=(popularity_var, "count"))
            .reset_index()
        )

    else:
        print("Select a correct variable: topic_documents, equal_size or popularity")

    nested_topics = nested_topics[lemma].drop_duplicates()
    nested_topics = pd.merge(nested_topics, group, on="lemma_2")
    nested_topics = nested_topics.sort_values("lemma_2")

    if count_search is not None:
        nested_topics = pd.merge(count_search, nested_topics, on="lemma_2", how="right")
        nested_topics = nested_topics.fillna(0)

    # Prepare in case there is a score to define the colors
    if "score_norm" in set(nested_topics.columns):
        color = "score_norm"
        color_continuous_midpoint = np.average(nested_topics[color])
    else:
        color = None
        color_continuous_midpoint = None

    path = [px.Constant("Topics Treemap")] + lemma

    if map_type == "treemap":
        fig = px.treemap(
            nested_topics,
            path=path,
            values="topic_size",
            color=color,
            color_continuous_scale="Picnic",
            color_continuous_midpoint=color_continuous_midpoint,
            width=width,
            height=height,
        )
    elif map_type == "sunburst":
        fig = px.sunburst(
            nested_topics,
            path=path,
            values="topic_size",
            color=color,
            color_continuous_scale="Picnic",
            color_continuous_midpoint=color_continuous_midpoint,
            width=width,
            height=height,
        )

    fig.update_traces(root_color="lightgrey")

    fig.data[
        0
    ].hovertemplate = (
        "labels=%{label}<br>topic_size=%{value}<br>parent=%{parent}<extra></extra>"
    )

    return fig
