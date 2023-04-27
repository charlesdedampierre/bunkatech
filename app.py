import pandas as pd
import streamlit as st
from bunkatech.bunka_class import Bunka
import toml
from PIL import Image

pd.options.mode.chained_assignment = None

instructions = toml.load("streamlit_instructions.toml")

st.set_page_config(
    layout="wide",
    page_icon=":)",
    initial_sidebar_state="collapsed",
    page_title="Bunka",
)
image = Image.open("images/bunka_logo.png")
image_2 = Image.open("images/shapingAI.png")
st.sidebar.image(image)
st.sidebar.image(image_2)
st.sidebar.title("")
st.sidebar.title("")

data = pd.read_csv("data/demo_data/data_minus_neutre_sample_3000_42.csv")
data = data.sample(3000, random_state=42)

terms_path = "data/demo_data/terms.csv"
terms_embeddings_path = "data/demo_data/terms_embeddings.csv"
docs_embeddings_path = "data/demo_data/docs_embeddings.csv"


@st.cache(allow_output_mutation=True)
def bunka_fit(data):
    bunka = Bunka(
        data=data,
        text_var="content",
        index_var="unique_id",
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
        terms_path=terms_path,
        terms_embeddings_path=terms_embeddings_path,
        docs_embeddings_path=docs_embeddings_path,
        docs_dimension_reduction=5,
    )
    bunka.fit(date_var="date", folding=None, popularity_var=None)
    return bunka


bunka = bunka_fit(data)

module_choice = st.sidebar.selectbox(
    label="Chose a Module",
    options=[
        "Semantic Origamis",
        "Semantic Nestedness",
        "Semantic Networks",
        "Top Documents",
        "Bourdieu Projection",
        "Temporal Trends",
        "Deep Dive",
    ],
    index=1,
)

if module_choice == "Semantic Origamis":
    # reset everytime we do this process
    bunka.data = bunka.data_initial.copy()
    st.title("Semantic Origamis")
    col1, col2 = st.columns([1, 3])
    with col1:
        left_axis = st.text_input(
            label="Enter the terms to describe the left part of the axis",
            value="positif, bien, rassurant",
        )
        right_axis = st.text_input(
            label="Enter the terms to describe the right part of the axis",
            value="négatif, mal, angoissant",
        )
        projection_item = st.selectbox(
            label="Chose what item to display", options=["terms", "documents"], index=1
        )
    with col2:
        if left_axis and right_axis:
            # Create a way to enter different words
            with st.spinner("Processing the request...."):
                fig = bunka.origami_projection_unique(
                    left_axis=left_axis.split(","),
                    right_axis=right_axis.split(","),
                    height=500,
                    width=1000,
                    type=projection_item,
                    dispersion=True,
                    barometer=True,
                )

                st.plotly_chart(fig)

if module_choice == "Semantic Nestedness":
    # reset everytime we do this process
    bunka.data = bunka.data_initial.copy()
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
            label="Explore the map by searching for terms", placeholder="Write a Query",
        )

        if query_input:
            query = query_input.split(",")
        else:
            query = None

    with col2:
        with st.spinner("Processing the request...."):

            fig = bunka.nested_maps(
                size_rule=map_size,
                map_type=map_type,
                width=1000,
                height=1000,
                query=query,
            )
            st.plotly_chart(fig)


if module_choice == "Semantic Networks":
    # reset everytime we do this process
    bunka.data = bunka.data_initial.copy()
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
        with st.spinner("Processing the request...."):
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

if module_choice == "Temporal Trends":
    # reset everytime we do this process
    with st.spinner("Processing the request...."):
        bunka.data = bunka.data_initial.copy()
        st.subheader("Temporal Trends")
        fig = bunka.temporal_topics_nested(date_var="date", normalize_y=False)
        st.plotly_chart(fig)

if module_choice == "Bourdieu Projection":
    # reset everytime we do this process
    st.subheader("Bourdieu Projection")
    col1, col2 = st.columns([1, 3])
    with col1:
        h_axis = st.text_input(
            label="Enter the terms to describe the horizontal axis",
            value="past, future",
        )
        v_axis = st.text_input(
            label="Enter the terms to describe the vertical axis", value="good, bad"
        )

        type = st.selectbox(
            label="Chose the Level of granularity",
            options=["terms", "documents"],
            index=0,
        )
        regression = st.selectbox(
            label="Display Linear Regression", options=["False", "True"], index=0
        )
    with col2:
        with st.spinner("Processing the request...."):

            fig = bunka.origami_projection(
                projection_1=h_axis.split(","),
                projection_2=v_axis.split(","),
                height=1000,
                width=1000,
                regression=regression,
                type=type,
            )

            st.plotly_chart(fig)


if module_choice == "Top Documents":
    # reset everytime we do this process
    bunka.data = bunka.data_initial.copy()
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
            "Sample Documents",
            "Top Terms",
        ],
        index=1,
    )

    if module_choice_deep_dive == "Sample Documents":
        st.subheader("Sample Documents")
        # top_docs = bunka.get_top_popular_docs(top_n=10)
        sample_data = bunka.data.sample(10)
        docs = sample_data[bunka.text_var]
        # docs = top_docs[bunka.text_var]
        # popularity = top_docs[bunka.popularity_var]
        for doc in docs:
            st.info(doc)
            # st.success("popularity score: " + str(pop))
            # st.text("")

    if module_choice_deep_dive == "Semantic Origamis":
        st.title("Semantic Origamis")
        col1, col2 = st.columns([1, 3])
        with col1:
            left_axis = st.text_input(
                label="Enter the terms to describe the left part of the axis",
                value="positif, bien, rassurant",
            )
            right_axis = st.text_input(
                label="Enter the terms to describe the right part of the axis",
                value="négatif, mal, angoissant",
            )

            projection_item = st.selectbox(
                label="Chose what item to display",
                options=["terms", "documents"],
                index=1,
            )
        with col2:
            if right_axis and left_axis:
                # Create a way to enter different words
                with st.spinner("Processing the request...."):
                    fig = bunka.origami_projection_unique(
                        left_axis=left_axis.split(","),
                        right_axis=right_axis.split(","),
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
            with st.spinner("Processing the request...."):
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

    if module_choice_deep_dive == "Top Terms":
        st.title("Top Terms")
        terms = bunka.df_indexed.reset_index(drop=True)
        terms = terms[["main form"]].drop_duplicates().reset_index(drop=True).head(100)
        st.dataframe(terms, height=1000)

    if module_choice == "Bourdieu Projection":
        print("Hello")
        # reset everytime we do this process
        st.subheader("Bourdieu Projection")
        col1, col2 = st.columns([1, 3])
        with col1:
            h_axis = st.text_input(
                label="Enter the terms to describe the horizontal axis",
                value="past, future",
            )
            v_axis = st.text_input(
                label="Enter the terms to describe the vertical axis", value="good, bad"
            )

            type = st.selectbox(
                label="Chose the Level of granularity",
                options=["terms", "documents"],
                index=0,
            )
            regression = st.selectbox(
                label="Display Linear Regression", options=["False", "True"], index=0
            )
        with col2:
            with st.spinner("Processing the request...."):

                fig = bunka.origami_projection(
                    projection_1=h_axis.split(","),
                    projection_2=v_axis.split(","),
                    height=1000,
                    width=1000,
                    regression=regression,
                    type=type,
                )

                st.plotly_chart(fig)
