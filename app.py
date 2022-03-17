import pandas as pd
from bunkatech.semantics.origami import Origami
from bunkatech.topic_modeling.nested_topics import NestedTopicModeling
from bunkatech.networks.networks_class import SemanticNetwork


import streamlit as st

pd.options.mode.chained_assignment = None

st.set_page_config(
    layout="wide",
    page_icon=":)",
    initial_sidebar_state="collapsed",
    page_title="Bunka",
)

"""
information about caching in streamlit:

- 'allow_output_mutation' makes it possible for the class to 
change its variable or methods.

- In order to cache the data,
the function needs to have the exact same input.If the data is sampled, then the function
reruns completely...

"""


@st.cache(allow_output_mutation=True)
def origami_fit(data):
    origami = Origami(data=data, text_var="description", index_var="imdb")

    origami.fit(
        extract_terms=True,
        docs_embedding=False,
        terms_embedding=True,
        sample_size_terms=100,
        terms_limit=100,
        terms_ents=False,
        terms_ngrams=(2, 2),
        terms_ncs=False,
        language="en",
        terms_embedding_model="/Volumes/OutFriend/sbert_model/distiluse-base-multilingual-cased-v1",
    )
    return origami


@st.cache(allow_output_mutation=True)
def network_fit(data):
    net = SemanticNetwork(data, text_var="description", index_var="imdb")
    net.fit(
        extract_terms=True,
        docs_embedding=False,
        terms_embedding=False,
        sample_size_terms=2000,
        terms_limit=1000,
        terms_ents=False,
        terms_ngrams=(2, 2),
        terms_ncs=True,
        terms_include_pos=["NOUN", "PROPN", "ADJ"],
        terms_include_types=["PERSON", "ORG"],
        terms_embedding_model="distiluse-base-multilingual-cased-v1",
        docs_embedding_model="distiluse-base-multilingual-cased-v1",
        language="en",
    )
    return net


@st.cache(allow_output_mutation=True)
def bunka_nested_fit(data):
    nested = NestedTopicModeling(data=data, text_var="description", index_var="imdb")
    nested.fit(
        folding=None,
        docs_embedding_model="tfidf",
        extract_terms=True,
        terms_embedding=False,
        sample_size_terms=3000,
        terms_limit=3000,
        terms_ents=False,
        terms_ngrams=(2, 2),
        terms_ncs=False,
        terms_include_pos=["NOUN", "PROPN", "ADJ"],
        terms_include_types=["PERSON", "ORG"],
        terms_embedding_model="distiluse-base-multilingual-cased-v1",
        language="en",
    )
    return nested


st.title("Bunka")

data = pd.read_csv(
    "/Users/charlesdedampierre/Desktop/ENS Projects/imaginary-world/db_film_iw (2).csv",
    index_col=[0],
)

data = data.sample(1000, random_state=42)

if st.checkbox("Origami Semantics"):
    origami = origami_fit(data)
    search = st.text_input(label="", placeholder="Enter the left end of the axis")

    if search:
        # Create a way to enter different words
        fig = origami.origami_projection_unique(
            projection_1=search.split(","),
            type="documents",
            height=500,
            width=1000,
            dispersion=True,
            barometer=True,
        )

        st.plotly_chart(fig)

if st.checkbox("Holistic Semantics"):

    nested = bunka_nested_fit(data)
    map_type = st.text_input(
        label="treemap", placeholder="enter 'Treemap' or 'Sunburst'"
    )

    query_input = st.text_input(label="", placeholder="Write a Query")

    if query_input:
        query = [query_input]
    else:
        query = None

    if map_type:
        fig = nested.nested_maps(
            size_rule="docs_size",
            map_type=map_type,
            width=1000,
            height=1000,
            query=query,
        )

        st.plotly_chart(fig)


if st.checkbox("Network Semantics"):
    net = network_fit(data)
    fig = net.pipeline(
        top_n=500,
        variables=["main form"],
        global_filter=0.2,
        n_neighbours=8,
        method="force_directed",
        n_cluster=15,
        bin_number=30,
        black_hole_force=5,
        color="community",
        size="size",
        symbol="entity",
        textfont_size=9,
        edge_size=0.5,
        height=1000,
        width=1000,
        template="plotly_dark",
    )

    st.plotly_chart(fig)
