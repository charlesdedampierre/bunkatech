import pandas as pd
import numpy as np
import plotly.express as px


def topics_treemap(
    nested_topics: pd.DataFrame,
    size="topic_size",
    max_depth=2,
    width=1000,
    height=1000,
):

    group = (
        nested_topics.groupby("lemma_2").agg(topic_size=("imdb", "count")).reset_index()
    )

    lemma = [f"lemma_{x}" for x in range(max_depth + 1)]
    nested_topics = nested_topics[lemma].drop_duplicates()

    nested_topics = pd.merge(nested_topics, group, on="lemma_2")
    nested_topics = nested_topics.sort_values("lemma_2")

    path = [px.Constant("Topics Treemap")] + lemma

    # Prepare in case there is a score to define the colors
    if "score_norm" in set(nested_topics.columns):
        color = "score_norm"
        color_continuous_midpoint = np.average(nested_topics[color])
    else:
        color = None
        color_continuous_midpoint = None

    fig = px.treemap(
        nested_topics,
        path=path,
        values=size,
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
