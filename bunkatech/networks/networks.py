from .networks_computation import (
    coocurrence_multiple_2,
    weight_to_similarity,
    compute_network,
)
from .networks_graph import draw_network
import pandas as pd


def network_analysis(
    df,
    variables=["genre", "country"],
    key="imdb",
    top_nodes=200,
    global_filter=0.4,
    n_neighbours=5,
    method="node2vec",
    density=4,
    height_att=4000,
    width_att=4000,
    color="community",
    template="simple_white",
    edge_size=1.5,
    symbol="entity",
    save_path=".",
):

    filename = (
        "_".join(variables)
        + "_method_"
        + method
        + "_global_filter_"
        + str(global_filter)
        + "_n_neighbours_"
        + str(n_neighbours)
        + "_top_nodes_"
        + str(top_nodes)
    )

    edges, nodes_attr = coocurrence_multiple_2(df, variables, key, top_n=top_nodes)
    edges_sim = weight_to_similarity(
        edges, global_filter=global_filter, n_neighbours=n_neighbours
    )

    G, pos_ = compute_network(edges_sim, method=method, density=density)

    fig = draw_network(
        G,
        pos_,
        nodes_attr,
        color=color,
        edge_size=edge_size,
        height_att=height_att,
        width_att=width_att,
        filename=save_path + "/" + filename,
        template=template,
        symbol=symbol,
    )

    return fig
