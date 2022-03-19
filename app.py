import pandas as pd
from bunkatech.bunka_class import Bunka
import streamlit as st
import toml
from PIL import Image

pd.options.mode.chained_assignment = None


instructions = toml.load("instructions.toml")


st.set_page_config(
    layout="wide",
    page_icon=":)",
    initial_sidebar_state="collapsed",
    page_title="Bunka",
)
image = Image.open(
    "/Users/charlesdedampierre/Desktop/BUNKATECH/topic_view/images/bunka_logo.png"
)
st.sidebar.image(image)


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
        terms_ents=False,
        terms_ngrams=(2, 2),
        terms_ncs=False,
        terms_include_pos=["NOUN", "PROPN", "ADJ"],
        terms_include_types=["PERSON", "ORG"],
        terms_embedding_model="distiluse-base-multilingual-cased-v1",
        docs_embedding_model="tfidf",
        language="en",
        terms_path=None,
        terms_embeddings_path=None,
        docs_embeddings_path=None,
        docs_dimension_reduction=5,
    )
    bunka.fit(date_var="year", folding=None, popularity_var="avg_vote")
    return bunka


data = pd.read_csv(
    "/Users/charlesdedampierre/Desktop/ENS Projects/imaginary-world/db_film_iw (2).csv",
    index_col=[0],
)

data = data.sample(1000, random_state=42)
bunka = bunka_fit(data)

module_choice = st.sidebar.selectbox(
    label="Chose a Module",
    options=[
        "Semantic Origamis",
        "Semantic Nestedness",
        "Semantic Networks",
        "Top Documents",
        "Deep Dive",
    ],
    index=1,
)

if module_choice == "Semantic Origamis":
    st.title("Semantic Origamis")
    col1, col2 = st.columns([1, 3])
    with col1:
        search = st.text_input(label="Enter a dimention", value="critique, promesse")
        projection_item = st.selectbox(
            label="Chose what item to display", options=["terms", "documents"], index=1
        )
    with col2:
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

if module_choice == "Semantic Nestedness":
    st.title("Semantic Nestedness")
    col1, col2 = st.columns([1, 3])
    with col1:
        map_type = st.selectbox(
            label="Chose the Mapping Type",
            options=["treemap", "sunburst", "sankey", "icicle"],
            index=0,
        )

        map_size = st.selectbox(
            label="Chose the Mapping Size",
            options=["equal_size", "docs_size"],
            index=0,
        )

        query_input = st.text_input(
            label="Explore the map by searching for terms",
            placeholder="Write a Query",
        )

        if query_input:
            query = query_input.split(",")
        else:
            query = None

    with col2:
        fig = bunka.nested_maps(
            size_rule=map_size,
            map_type=map_type,
            width=1000,
            height=1000,
            query=query,
        )

        st.plotly_chart(fig)


if module_choice == "Semantic Networks":
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


if module_choice == "Top Documents":
    st.subheader("Top Documents")
    top_docs = bunka.get_top_popular_docs(top_n=10)
    docs = top_docs[bunka.text_var]
    popularity = top_docs[bunka.popularity_var]
    for doc, pop in zip(docs, popularity):
        st.info(doc)
        st.success("popularity score: " + str(pop))
        st.text("")


if module_choice == "Deep Dive":

    # First Level
    level_0_list = list(set(bunka.h_clusters_names["lemma_0"].to_list())) + ["None"]

    default_ix = level_0_list.index("None")
    lv_0 = st.selectbox("Chose the Level 1", options=level_0_list, index=default_ix)

    # Second Level If None, then impossible to chose sub-topic
    if lv_0 == "None":
        level_1_list = ["None"]
    else:
        mask_1 = bunka.h_clusters_names["lemma_0"] == lv_0
        level_1_list = list(
            set(bunka.h_clusters_names[mask_1]["lemma_1"].to_list())
        ) + ["None"]

    default_ix = level_1_list.index("None")
    lv_1 = st.selectbox("Chose the Level 2", options=level_1_list, index=default_ix)

    # If None, then impossible to chose sub-topic in the selectbox
    if lv_1 == "None" or lv_0 == "None":
        level_2_list = ["None"]
    else:
        mask_2 = bunka.h_clusters_names["lemma_1"] == lv_1
        level_2_list = list(
            set(bunka.h_clusters_names[mask_2]["lemma_2"].to_list())
        ) + ["None"]

    default_ix = level_2_list.index("None")
    lv_2 = st.selectbox("Chose the Level 3", options=level_2_list, index=default_ix)

    if lv_0 == "None":
        lv_0 = None
    if lv_1 == "None":
        lv_1 = None
    if lv_2 == "None":
        lv_2 = None

    bunka.filter_nested_cluster(lv_0=lv_0, lv_1=lv_1, lv_2=lv_2)

    module_choice_deep_dive = st.sidebar.selectbox(
        label="Chose a Module to deep dive in:",
        options=[
            "Semantic Origamis",
            "Semantic Networks",
            "Top Documents",
        ],
        index=1,
    )

    if module_choice_deep_dive == "Top Documents":
        st.subheader("Top Documents")
        top_docs = bunka.get_top_popular_docs(top_n=10)
        docs = top_docs[bunka.text_var]
        popularity = top_docs[bunka.popularity_var]
        for doc, pop in zip(docs, popularity):
            st.info(doc)
            st.success("popularity score: " + str(pop))
            st.text("")

    if module_choice_deep_dive == "Semantic Origamis":
        st.title("Semantic Origamis")
        col1, col2 = st.columns([1, 3])
        with col1:
            search = st.text_input(
                label="Enter a dimention", value="critique, promesse"
            )
            projection_item = st.selectbox(
                label="Chose what item to display",
                options=["terms", "documents"],
                index=1,
            )
        with col2:
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

    if module_choice_deep_dive == "Semantic Networks":
        st.title("Semantic Networks")
        col1, col2 = st.columns([1, 3])
        with col1:
            top_n = st.number_input(label="Top nodes", value=50)
            top_n = int(top_n)
            black_hole_force = st.number_input(label="Black Hole Force", value=1)
            n_cluster = st.number_input(label="Cluster Number", value=10)
            global_filter = st.number_input(label="global_filter", value=0.2)
            n_neighbours = st.number_input(label="n_neighbours", value=10)

        with col2:
            fig = bunka.fit_draw(
                variables=["main form"],
                top_n=top_n,
                global_filter=global_filter,
                n_neighbours=n_neighbours,
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
