import networkx as nx
import numpy as np
import pandas as pd
import multiprocessing
from collections import Counter
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from node2vec import Node2Vec
import plotly.graph_objects as go
from ..basic_class import BasicSemantics


class SemanticNetworks(BasicSemantics):
    def __init__(
        self,
        data,
        text_var,
        index_var,
        extract_terms=True,
        terms_embedding=True,
        docs_embedding=True,
        sample_size_terms=500,
        terms_limit=500,
        terms_ents=True,
        terms_ngrams=(1, 2),
        terms_ncs=True,
        terms_include_pos=["NOUN", "PROPN", "ADJ"],
        terms_include_types=["PERSON", "ORG"],
        terms_embedding_model="distiluse-base-multilingual-cased-v1",
        docs_embedding_model="tfidf",
        language="en",
        terms_path=None,
        terms_embeddings_path=None,
        docs_embeddings_path=None,
    ) -> None:

        BasicSemantics.__init__(
            self,
            data=data,
            text_var=text_var,
            index_var=index_var,
            terms_path=terms_path,
            terms_embeddings_path=terms_embeddings_path,
            docs_embeddings_path=docs_embeddings_path,
        )

        BasicSemantics.fit(
            self,
            extract_terms=extract_terms,
            terms_embedding=terms_embedding,
            docs_embedding=docs_embedding,
            sample_size_terms=sample_size_terms,
            terms_limit=terms_limit,
            terms_ents=terms_ents,
            terms_ngrams=terms_ngrams,
            terms_ncs=terms_ncs,
            terms_include_pos=terms_include_pos,
            terms_include_types=terms_include_types,
            terms_embedding_model=terms_embedding_model,
            docs_embedding_model=docs_embedding_model,
            language=language,
        )

    def fit_draw(
        self,
        variables,
        top_n=100,
        global_filter=0.2,
        n_neighbours=6,
        method="node2vec",
        n_cluster=10,
        bin_number=30,
        black_hole_force=1.5,
        color="community",
        size="size",
        symbol="entity",
        textfont_size=9,
        edge_size=1,
        height=1000,
        width=1000,
        template="plotly_dark",
    ):
        self.data_network = pd.merge(
            self.data, self.df_indexed.reset_index(), on=self.index_var
        )

        self.variables = variables
        self.coocurrence_multiple()
        self.get_top_nodes(top_n=top_n)
        self.weight_to_similarity(
            global_filter=global_filter, n_neighbours=n_neighbours
        )

        self.compute_network(
            density=black_hole_force,
            bin_number=bin_number,
            method=method,
            n_cluster=n_cluster,
        )

        fig = self.draw_network(
            color=color,
            size=size,
            symbol=symbol,
            textfont_size=textfont_size,
            edge_size=edge_size,
            height_att=height,
            width_att=width,
            template=template,
        )

        return fig

    def coocurrence_multiple(self):

        fin = pd.DataFrame()
        for var in self.variables:
            df_var = self.data_network[[self.index_var, var]]
            df_var = df_var.drop_duplicates()
            df_var = (
                df_var.groupby([self.index_var, var])[self.index_var]
                .count()
                .rename("weight")
                .reset_index()
            )
            df_var = df_var.rename(columns={var: "data"})
            df_var["entity"] = var
            fin = fin.append(df_var)

        self.fin = fin

        df_co = pd.merge(self.fin, self.fin, on=self.index_var)
        df_co["product"] = df_co["weight_x"] * df_co["weight_y"]
        self.edges = df_co.groupby(["data_x", "data_y"])["product"].sum().reset_index()
        self.edges.columns = ["source", "target", "weight"]

        return self.edges

    def get_top_nodes(self, top_n=300):

        # filter by top_n regarding their size
        nodes_attr = self.fin[[self.index_var, "data", "entity"]]
        nodes_attr = (
            nodes_attr.groupby(["data", "entity"])[self.index_var]
            .count()
            .rename("size")
            .reset_index()
        )
        nodes_attr = nodes_attr.sort_values("size", ascending=False).reset_index(
            drop=True
        )
        self.nodes_attr = nodes_attr.head(top_n)
        self.edges = self.edges[
            self.edges.source.isin(list(self.nodes_attr["data"]))
            & self.edges.target.isin(list(self.nodes_attr["data"]))
        ]

        return self.edges

    def weight_to_similarity(self, global_filter: float = 0.7, n_neighbours: int = 6):

        """

        This functions transform an edge list with weights into an edge list whose weights are cosine similarity

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

        pivot = self.edges.pivot("source", "target", "weight")
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
        self.new_edge = new_edge.reset_index(drop=True)

        return self.new_edge

    def compute_network(
        self,
        density: int = 2,
        bin_number: int = 30,
        method: str = "node2vec",
        n_cluster: int = 10,
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

        def add_black_holes(G, density=2):

            # Add centroids in the middle of every 'community' and connects it to
            # all the nodes in the 'community'

            df_node = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
            for community_number in set(df_node.community):

                G.add_node(
                    f"network_center_{community_number}",
                    community=community_number,
                    size=1,
                    entity="centroid",
                )

                list_nodes = list(
                    df_node[df_node["community"] == community_number].index
                )

                for node in list_nodes:
                    G.add_edge(
                        f"network_center_{community_number}", node, weight=density
                    )

            return G

        # Create Graph Object
        G = nx.from_pandas_edgelist(
            self.new_edge, source="source", target="target", edge_attr="weight"
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

        # add centrality attribute to the G object
        node_attr_centrality = centrality.to_dict("index")
        nx.set_node_attributes(G, node_attr_centrality)

        if method == "force_directed":
            # Create the laout with Fruchterman-Reingold force-directed algorithm
            # Louvain community
            partition = community_louvain.best_partition(G)
            partition = pd.DataFrame.from_dict(
                partition, orient="index", columns=["community"]
            )

            # Add community attribute to the G object
            node_attr_community = partition.to_dict("index")
            nx.set_node_attributes(G, node_attr_community)
            G = add_black_holes(G, density=density)

            # Compute the coordinate of nodes based on specific force algorithm
            pos_ = nx.spring_layout(G)

            self.df_embeddings = pd.DataFrame(pos_).T
            self.df_embeddings.index = G.nodes()

        elif method == "node2vec":

            node2vec = Node2Vec(G, dimensions=700, workers=multiprocessing.cpu_count())
            model = node2vec.fit(window=30, min_count=1)
            nodes = list(map(str, G.nodes()))
            embeddings = np.array([model.wv[x] for x in nodes])

            # Get the community with embeddings
            cluster_model = KMeans(n_clusters=n_cluster)
            community = cluster_model.fit_predict(embeddings)

            partition = pd.DataFrame(index=nodes)
            partition["community"] = community

            # Add community attribute to the G object
            node_attr_community = partition.to_dict("index")
            nx.set_node_attributes(G, node_attr_community)
            G = add_black_holes(G, density=density)

            # Re-compute to add the new nodes and get their
            node2vec = Node2Vec(G, dimensions=700, workers=multiprocessing.cpu_count())
            model = node2vec.fit(window=30, min_count=1)
            nodes = list(map(str, G.nodes()))
            embeddings = np.array([model.wv[x] for x in nodes])

            # Get the 2D embeddings to display data
            tsne = TSNE(n_components=2)
            embeddings = tsne.fit_transform(embeddings)
            pos_ = {nodes[x]: embeddings[x] for x in range(len(nodes))}

            self.df_embeddings = pd.DataFrame(embeddings)
            self.df_embeddings.index = nodes

        self.G = G
        self.pos_ = pos_

        self.df_node = pd.DataFrame.from_dict(
            dict(self.G.nodes(data=True)), orient="index"
        )

    def draw_network(
        self,
        color="entity",
        size="size",
        symbol=None,
        textfont_size=9,
        edge_size=3,
        height_att=1000,
        width_att=1000,
        template="plotly_dark",
    ):
        """Output a Graph

        Args:
            color (str, optional): chose in the nodes attribute the column_name for colors. Defaults to "entity".
            size (str, optional): chose in the nodes attribute the column_name for size. Defaults to "size".
            symbol ([type], optional): [description]. Defaults to None.
        """

        # Deal the size of the centroids (as they do not come from the nodes_attr)
        clusters = [
            self.G.nodes(data=True)[x]["community"] for x in list(self.G.nodes())
        ]
        clusters = dict(Counter(clusters))

        # Add the entities and the size
        df_nodes = self.nodes_attr.set_index("data")
        df_nodes["entity"] = df_nodes["entity"].astype("category").cat.codes
        bin_number = 30
        df_nodes["size"] = pd.cut(
            df_nodes["size"].rank(method="first"),
            bin_number,
            labels=range(1, bin_number + 1),
        )

        node_attr = df_nodes.to_dict("index")
        nx.set_node_attributes(self.G, node_attr)

        self.df_node = pd.DataFrame.from_dict(
            dict(self.G.nodes(data=True)), orient="index"
        )
        self.df_node = pd.merge(
            self.df_embeddings, self.df_node, left_index=True, right_index=True
        )

        # For each edge, make an edge_trace, append to list
        edge_trace = []
        for edge in self.G.edges():

            if self.G.edges()[edge]["weight"] > 0:
                x0, y0 = self.pos_[edge[0]]
                x1, y1 = self.pos_[edge[1]]

                """text = (
                    char_1 + "--" + char_2 + ": " + str(self.G.edges()[edge]["weight"])
                )"""

                if "network_center_" in edge[0]:
                    width = 0
                elif "network_center_" in edge[1]:
                    width = 0

                else:
                    # The bigger the node, the bigger the edge width
                    width = edge_size * self.G.edges()[edge]["weight"] ** 1.75

                trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=width, color="cornflowerblue"),
                    mode="lines",
                )
                # fig.add_trace(trace)
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

        # For each nodeget the position and size and add to the node_trace
        for node in self.G.nodes():

            if "network_center_" in node:
                continue

            x, y = self.pos_[node]
            node_trace["x"] += tuple([x])
            node_trace["y"] += tuple([y])
            node_trace["marker"]["color"] += tuple([self.G.nodes()[node][color]])

            if symbol is None:
                node_trace["marker"]["symbol"] = "circle"
            else:
                node_trace["marker"]["symbol"] += tuple([self.G.nodes()[node][symbol]])

            node_trace["text"] += tuple(["<b>" + node + "</b>"])
            node_trace["marker"]["opacity"] += tuple([0.6])
            node_trace["marker"]["size"] += tuple([self.G.nodes()[node][size]])

        # Customize layout
        layout = go.Layout(
            height=height_att,
            width=width_att,
            title="Semantic Network",
            xaxis={"showgrid": False, "zeroline": False},  # no gridlines
            yaxis={"showgrid": False, "zeroline": False},  # no gridlines
        )

        # Create figure
        fig = go.Figure(layout=layout)

        # Add the traces
        for trace in edge_trace:
            fig.add_trace(trace)

        fig.add_trace(node_trace)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(template=template)

        return fig
