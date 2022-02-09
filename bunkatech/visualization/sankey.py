import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly
import plotly.express as px


def flatten(list):
    return [item for sublist in list for item in sublist]


def make_sankey(df: pd.DataFrame, field: str, index_var: str):

    # first source/values
    source_initital = [0 for x in np.arange(1, 9)]
    target_initital = [x for x in np.arange(1, 9)]
    values_initial = []

    values_initial = (
        df.groupby("level_0", sort=False)
        .agg(values=(index_var, "count"))["values"]
        .to_list()
    )

    source_0 = [[x] * 4 for x in np.arange(1, 9)]
    source_0 = flatten(source_0)

    target_0 = df["level_1"].drop_duplicates().tolist()
    target_0 = [x + 9 for x in target_0]
    values_0 = (
        df.groupby("level_1", sort=False)
        .agg(values=(index_var, "count"))["values"]
        .to_list()
    )

    source_1 = [[x] * 2 for x in np.arange(9, 41)]
    source_1 = flatten(source_1)

    target_1 = df["level_2"].drop_duplicates().tolist()
    target_1 = [x + 41 for x in target_1]
    values_1 = (
        df.groupby("level_2", sort=False)
        .agg(values=(index_var, "count"))["values"]
        .to_list()
    )

    names_initial = [field]
    names_0 = df["lemma_0"].drop_duplicates().tolist()
    names_1 = df["lemma_1"].drop_duplicates().tolist()
    names_2 = df["lemma_2"].drop_duplicates().tolist()

    color_initial = ["red"]
    color_0 = ["magenta" for col in range(len(names_0))]
    color_1 = ["green" for col in range(len(names_1))]
    color_2 = ["blue" for col in range(len(names_2))]

    colors = color_initial + color_0 + color_1 + color_2

    # final
    names = names_initial + names_0 + names_1 + names_2
    sources = source_initital + source_0 + source_1
    targets = target_initital + target_0 + target_1
    values = values_initial + values_0 + values_1

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=100,
                    thickness=30,
                    line=dict(color="black", width=0.5),
                    label=names,
                    # color="blue",
                    color=colors,
                ),
                link=dict(
                    source=sources,  # indices correspond to labels, eg A1, A2, A1, B1, ...
                    target=targets,
                    value=values,
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Basic Sankey Diagram",
        font_size=16,
        width=4000,
        height=2500,
        margin=dict(l=50, r=800, b=50, t=50, pad=4),
    )

    return fig


"""project_path = "/Volumes/OutFriend/bunkamap_examples/political_psychology"
storage_path = project_path + "/bunkamap"
df = pd.read_csv(storage_path + "/df_clusters_name.csv")


fig = make_sankey(df, field="Political Psychology")
plotly.offline.plot(fig, auto_open=False, filename=project_path + "/sankey.html")

fig.show()


print(df)
"""
