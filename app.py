import pandas as pd
from bunkatech.bunka_class import Bunka
import streamlit as st
import toml

instructions = toml.load("instructions.toml")


pd.options.mode.chained_assignment = None

st.set_page_config(
    layout="wide",
    page_icon=":)",
    initial_sidebar_state="collapsed",
    page_title="Bunka",
)

st.write(instructions["app"]["app_intro"])


@st.cache(allow_output_mutation=True)
def bunka_fit(data):
    bunka = Bunka(
        data=data,
        text_var="description",
        index_var="imdb",
        extract_terms=True,
        terms_embedding=True,
        docs_embedding=True,
        sample_size_terms=2000,
        terms_limit=2000,
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
    )
    bunka.fit(date_var="year", folding=None, popularity_var="avg_vote")
    return bunka


st.sidebar.title("Bunka")

data = pd.read_csv(
    "/Users/charlesdedampierre/Desktop/ENS Projects/imaginary-world/db_film_iw (2).csv",
    index_col=[0],
)

data = data.sample(1000, random_state=42)
bunka = bunka_fit(data)
if st.sidebar.checkbox("Semantic Origamis"):
    st.title("Semantic Origamis")
    search = st.text_input(label="", placeholder="Enter the left end of the axis")
    projection_item = st.selectbox(
        label="Chose what item to display", options=["terms", "documents"], index=1
    )

    if search:
        # Create a way to enter different words
        fig = bunka.origami_projection_unique(
            projection_1=search.split(","),
            height=500,
            width=1000,
            type=projection_item,
            dispersion=True,
            barometer=True,
        )

        st.plotly_chart(fig)

if st.sidebar.checkbox("Semantic Nestedness"):
    st.title("Semantic Nestedness")
    map_type = st.selectbox(
        label="Chose the Mapping Type",
        options=["treemap", "sunburst", "sankey"],
        index=0,
    )

    query_input = st.text_input(
        label="Explore the map by searching for terms", placeholder="Write a Query"
    )

    if query_input:
        query = query_input.split(",")
    else:
        query = None

    if map_type:
        fig = bunka.nested_maps(
            size_rule="equal_size",
            map_type=map_type,
            width=1000,
            height=1000,
            query=query,
        )

        st.plotly_chart(fig)


if st.sidebar.checkbox("Semantic Networks"):
    st.title("Semantic Networks")
    col1, col2 = st.columns([1, 3])
    with col1:
        top_n = st.number_input(label="Top nodes", value=50)
        top_n = int(top_n)
        black_hole_force = st.number_input(label="Black Hole Force", value=1)
        n_cluster = st.number_input(label="Cluster Number", value=10)

    with col2:
        fig = bunka.fit_draw(
            variables=["main form"],
            top_n=top_n,
            global_filter=0.2,
            n_neighbours=6,
            method="force_directed",
            n_cluster=n_cluster,
            bin_number=30,
            black_hole_force=black_hole_force,
            color="community",
            size="size",
            symbol="entity",
            textfont_size=9,
            edge_size=1,
            height=1000,
            width=1000,
            template="plotly_dark",
        )

        st.plotly_chart(fig)


if st.sidebar.checkbox("Top Documents"):
    st.subheader("Top Documents")
    top_docs = bunka.get_top_popular_docs(top_n=10)
    docs = top_docs[bunka.text_var]
    popularity = top_docs[bunka.popularity_var]
    for doc, pop in zip(docs, popularity):
        st.info(doc)
        st.success("popularity score: " + str(pop))
        st.text("")
