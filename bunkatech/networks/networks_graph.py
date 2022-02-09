import plotly
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from collections import Counter


def make_edge(x, y, text, width):
    return go.Scatter(
        x=x,
        y=y,
        line=dict(width=width, color="cornflowerblue"),
        hoverinfo="text",
        text=([text]),
        mode="lines",
    )


def draw_network(
    G,
    pos_,
    nodes_attr,
    color="entity",
    size="size",
    symbol=None,
    textfont_size=9,
    edge_size=3,
    filename="test",
    height_att=4000,
    width_att=2000,
    template="plotly_dark",
):
    """Output a Graph

    Args:
        G ([type]): Networkx object
        pos_ ([type]): nodes coordinate
        nodes_attr ([type]): nodes attribute
        color (str, optional): chose in the nodes attribute the column_name for colors. Defaults to "entity".
        size (str, optional): chose in the nodes attribute the column_name for size. Defaults to "size".
        symbol ([type], optional): [description]. Defaults to None.
    """

    # Deal the size of the centroids (as they do not come from the nodes_attr)
    clusters = [G.nodes(data=True)[x]["community"] for x in list(G.nodes())]
    clusters = dict(Counter(clusters))

    for community_number, count_community in clusters.items():
        line = pd.DataFrame(
            {
                "data": [f"network_center_{community_number}"],
                "entity": ["centroid"],
                "size": [count_community],
            },
            index=[0],
        )

        nodes_attr = nodes_attr.append(line)

    nodes_attr = nodes_attr.reset_index(drop=True)

    entities = nodes_attr[["data", "entity"]]
    entities.index = entities.data
    entities = entities.drop("data", axis=1)
    entities["entity"] = entities["entity"].astype("category").cat.codes

    node_attr_entities = entities.to_dict("index")
    nx.set_node_attributes(G, node_attr_entities)

    size_attr = nodes_attr[["data", "size"]]
    bin_number = 30
    size_attr["size"] = pd.cut(
        size_attr["size"].rank(method="first"),
        bin_number,
        labels=range(1, bin_number + 1),
    )

    # Increase the size of the suns
    # size_attr[size_attr['entity'] == 'centroid']['size'] = size_attr[size_attr['entity'] == 'centroid']['size']*10

    size_attr["size"] = size_attr["size"].astype(int)
    size_attr.index = size_attr.data
    size_attr = size_attr.drop("data", axis=1)

    node_attr_size = size_attr.to_dict("index")
    nx.set_node_attributes(G, node_attr_size)

    # For each edge, make an edge_trace, append to list
    edge_trace = []
    for edge in G.edges():

        if G.edges()[edge]["weight"] > 0:
            char_1 = edge[0]
            char_2 = edge[1]
            x0, y0 = pos_[char_1]
            x1, y1 = pos_[char_2]
            text = char_1 + "--" + char_2 + ": " + str(G.edges()[edge]["weight"])

            # erase connections to center
            if "network_center_" in char_1:
                width = 0
            elif "network_center_" in char_2:
                width = 0
            else:
                width = edge_size * G.edges()[edge]["weight"] ** 1.75

            trace = make_edge([x0, x1, None], [y0, y1, None], text, width)
        edge_trace.append(trace)

    # Make a node trace
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        textposition="top center",
        textfont_size=textfont_size,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(color=[], size=[], line=None, opacity=[], symbol=[]),
    )

    # For each node in midsummer, get the position and size and add to the node_trace
    for node in G.nodes():
        x, y = pos_[node]
        node_trace["x"] += tuple([x])
        node_trace["y"] += tuple([y])
        node_trace["marker"]["color"] += tuple([G.nodes()[node][color]])

        if symbol is None:
            node_trace["marker"]["symbol"] = "circle"
        else:
            node_trace["marker"]["symbol"] += tuple([G.nodes()[node][symbol]])

        if "network_center_" in node:
            pass
            # node_trace["text"] += tuple()
            # node_trace["marker"]["opacity"] += tuple([0.08])
            # node_trace["marker"]["size"] += tuple([G.nodes()[node][size] * 10])
        else:
            node_trace["text"] += tuple(["<b>" + node + "</b>"])
            node_trace["marker"]["opacity"] += tuple([0.6])
            node_trace["marker"]["size"] += tuple([G.nodes()[node][size]])

    # Customize layout
    layout = go.Layout(
        height=height_att,
        width=width_att,
        # paper_bgcolor='rgba(0,0,0,0)', # transparent background
        # plot_bgcolor='rgba(0,0,0,0)', # transparent 2nd background
        xaxis={"showgrid": False, "zeroline": False},  # no gridlines
        yaxis={"showgrid": False, "zeroline": False},  # no gridlines
    )

    # Create figure
    fig = go.Figure(layout=layout)
    for trace in edge_trace:
        fig.add_trace(trace)

    fig.add_trace(node_trace)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(template=template)

    return fig
