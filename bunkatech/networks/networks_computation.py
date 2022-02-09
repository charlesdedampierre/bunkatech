import networkx as nx
import plotly.graph_objects as go
import numpy as np
import community as community_louvain
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
import multiprocessing
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def coocurrence_multiple_2(df, variables, key, top_n=300):

    fin = pd.DataFrame()
    for var in variables:
        df_var = df[[key, var]]
        df_var = df_var.drop_duplicates()
        df_var = df_var.groupby([key, var])[key].count().rename("weight").reset_index()
        df_var = df_var.rename(columns={var: "data"})
        df_var["entity"] = var
        fin = fin.append(df_var)

    nodes_attr = fin[[key, "data", "entity"]]
    nodes_attr = (
        nodes_attr.groupby(["data", "entity"])[key].count().rename("size").reset_index()
    )
    nodes_attr = nodes_attr.sort_values("size", ascending=False).reset_index(drop=True)
    nodes_attr = nodes_attr.head(top_n)

    fin = fin.drop_duplicates()

    df_co = pd.merge(fin, fin, on=key)
    df_co["product"] = df_co["weight_x"] * df_co["weight_y"]
    edges = df_co.groupby(["data_x", "data_y"])["product"].sum().reset_index()
    edges.columns = ["source", "target", "weight"]
    edges = edges[
        edges.source.isin(list(nodes_attr.data))
        & edges.target.isin(list(nodes_attr.data))
    ]

    return edges, nodes_attr


def coocurrence_multiple(conn, variables, key, top_n=300):

    df = pd.DataFrame()
    for var in variables:
        df_var = pd.read_sql_query(f"SELECT * FROM {var}", conn)
        if "weight" not in list(df_var.columns):
            df_var["weight"] = 1
        df_var = df_var[[key, "data", "weight"]]
        df_var["entity"] = var
        df = df.append(df_var)

    nodes_attr = df[[key, "data", "entity"]]
    nodes_attr = (
        nodes_attr.groupby(["data", "entity"])[key].count().rename("size").reset_index()
    )
    nodes_attr = nodes_attr.sort_values("size", ascending=False).reset_index(drop=True)
    nodes_attr = nodes_attr.head(top_n)

    df = df.drop_duplicates()

    df_co = pd.merge(df, df, on=key)
    df_co["product"] = df_co["weight_x"] * df_co["weight_y"]
    edges = df_co.groupby(["data_x", "data_y"])["product"].sum().reset_index()
    edges.columns = ["source", "target", "weight"]
    edges = edges[
        edges.source.isin(list(nodes_attr.data))
        & edges.target.isin(list(nodes_attr.data))
    ]

    return edges, nodes_attr


def weight_to_similarity(
    cooc_df: pd.DataFrame, global_filter: float = 0.7, n_neighbours: int = 6
):

    """

    This functions transform an edge list with weights into an edge list whose weights are cosinuse similarity

    parameters:
        - cooc_df with the following columns:
            - source
            - target
            - weight
        - global_filter: filter with minimum similarity
        - n_neighbours: filter with minimum neighbours

    output:
        - edge_list with the following columns:
            - source
            - target
            - weight (cosine similarity)
            - rank (closeness to the source)

    """

    pivot = cooc_df.pivot("source", "target", "weight")
    pivot = pivot.fillna(0)
    similarity = cosine_similarity(pivot)  # compute cosine similarity
    df_sim = pd.DataFrame(similarity, index=pivot.index, columns=pivot.columns)
    df_sim = df_sim[(df_sim >= global_filter)]
    df_sim["nodes"] = df_sim.index

    res_g = pd.melt(df_sim, id_vars=["nodes"]).sort_values("nodes")  # time
    res_g = res_g.dropna()
    res_g.columns = ["source", "target", "weight"]
    res_g = res_g.sort_values("source")

    # Erase duplicates
    duplicates = []
    for x, y, i in zip(res_g.source, res_g.target, res_g.index):
        if x == y:
            duplicates.append(i)

    new_edge = res_g.drop(index=duplicates)
    new_edge = new_edge[new_edge.weight != 0]

    # Filter Neighbours
    new_edge["rank"] = new_edge.groupby(["source"])["weight"].rank(
        method="first", ascending=False
    )
    new_edge = new_edge[new_edge["rank"] <= n_neighbours]
    new_edge = new_edge.reset_index(drop=True)

    return new_edge


def compute_network(
    edges: pd.DataFrame,
    density: int = 1,
    larger: int = 4,
    bin_number: int = 30,
    method: str = "node2vec",
    n_cluster: int = 20,
):
    """This function takes edges as input and outputs a Networkx Object 'G' and
    coordiate

    Args:
        density (int, optional): density of the clusters force force-directed algorithm only. Defaults to 1.
        larger (int, optional): larger of the nodes_size . Defaults to 4.
        bin_number (int, optional):  number of bins to normalize the nodes' size. Defaults to 30.
        method (str, optional): node2vec or force_directed algorithm. Defaults to "node2vec".
        n_cluster (int, optional): [description]. Defaults to 20.

    Returns:
        [type]: Networkx G object and an array of coordinate for all the nodes in the Networkx G
    """

    # Create Graph
    G = nx.Graph()
    G = nx.from_pandas_edgelist(
        edges, source="source", target="target", edge_attr="weight"
    )

    # Compute centrality
    centrality = nx.degree_centrality(G)
    centrality = pd.DataFrame.from_dict(
        centrality, orient="index", columns=["centrality"]
    )

    centrality["centrality"] = pd.qcut(
        centrality["centrality"].rank(method="first"),
        bin_number,
        labels=range(1, bin_number + 1),
    )

    # add centrality attribute
    node_attr_centrality = centrality.to_dict("index")
    nx.set_node_attributes(G, node_attr_centrality)

    if method == "force_directed":
        # Create the laout with Fruchterman-Reingold force-directed algorithm
        # Louvain community
        partition = community_louvain.best_partition(G)
        partition = pd.DataFrame.from_dict(
            partition, orient="index", columns=["community"]
        )

        # Add community attribute
        node_attr_community = partition.to_dict("index")
        nx.set_node_attributes(G, node_attr_community)

        # Add a node for every community and connect it to all the other nodes
        com = []
        for node in G.nodes():
            res = (node, G.nodes()[node]["community"])
            com.append(res)

        df_com = pd.DataFrame(com, columns=["node_name", "community"])

        for community_number in set(df_com.community):
            com = list(df_com[df_com.community == community_number].node_name)
            G.add_node(
                f"network_center_{community_number}",
                centrality=len(com) * larger,
                community=community_number,
                size=len(com) * bin_number,
            )

            for node_com in com:
                G.add_edge(
                    f"network_center_{community_number}", node_com, weight=density
                )

        pos_ = nx.spring_layout(G)

    elif method == "node2vec":

        node2vec = Node2Vec(G, dimensions=700, workers=multiprocessing.cpu_count())
        model = node2vec.fit(window=30, min_count=1)
        nodes = list(map(str, G.nodes()))
        embeddings = np.array([model.wv[x] for x in nodes])

        # Reduc to 5 dimention
        # tsne = TSNE(n_components=5, method='exact')
        # embeddings_reduc = tsne.fit_transform(embeddings)

        # Get the community with embeddings
        cluster_model = KMeans(n_clusters=n_cluster)
        # cluster_model = hdbscan.HDBSCAN(min_cluster_size=3,metric='euclidean', cluster_selection_method='eom')
        community = cluster_model.fit_predict(embeddings)

        partition = pd.DataFrame(index=nodes)
        partition["community"] = community

        # Add community attribute
        node_attr_community = partition.to_dict("index")
        nx.set_node_attributes(G, node_attr_community)

        tsne = TSNE(n_components=2)
        embeddings = tsne.fit_transform(embeddings)
        pos_ = {nodes[x]: embeddings[x] for x in range(len(nodes))}

    return G, pos_
