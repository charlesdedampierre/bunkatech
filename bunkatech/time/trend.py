import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_dark"


def trend(
    data,
    date_var="date",
    index_var="id",
    smoothing_scale=7,
    context_scale=182,
    height=1000,
    width=2000,
    stv_scale=1.5,
):

    """Count the number of documents in a specific date time"""

    df = data.groupby(date_var).agg(count_ids=(index_var, "count")).reset_index()
    df["trend"] = df["count_ids"].rolling(smoothing_scale).mean()
    df["long_trend"] = df["count_ids"].rolling(context_scale).mean()
    df["lower"] = (
        df["count_ids"].rolling(context_scale).mean()
        - df["count_ids"].rolling(context_scale).std() * stv_scale
    )
    df["upper"] = (
        df["count_ids"].rolling(context_scale).mean()
        + df["count_ids"].rolling(context_scale).std() * stv_scale
    )

    trace1 = go.Scatter(
        x=df[date_var],
        y=df["count_ids"],
        mode="lines",
        marker=dict(
            # color='rgb(0, 0, 0)',
        ),
        name="data",
    )

    trace2 = go.Scatter(
        x=df[date_var],
        y=df["trend"],
        mode="lines",
        marker=dict(
            # color='#5E88FC',
            # symbol='circle-open'
        ),
        name="moving_average",
    )

    trace3 = go.Scatter(
        x=df[date_var],
        y=df["lower"],
        mode="lines",
        marker=dict(
            color="grey",
            # symbol='circle-open'
        ),
        name="lower",
    )

    trace4 = go.Scatter(
        x=df[date_var],
        y=df["upper"],
        mode="lines",
        marker=dict(
            color="grey",
            # symbol='circle-open'
        ),
        name="higher",
        fill="tonexty",
    )

    layout = go.Layout(showlegend=True, height=500)

    data = [trace1, trace2, trace3, trace4]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        title=dict(
            text="<b>Semantic Candles</b>",
            font=dict(family="Arial", size=50, color="#000000"),
        ),
        height=height,
        width=width,
    )

    return fig


def semantic_candles(
    df,
    date_var="date",
    index_var="id",
    short_trend_step=5,
    long_trend_step=20,
    height=1000,
    width=2300,
    time_step="month",
):
    """Display the semantics as candle
    insert AAAA-MM-DD. The algorithmn groups by months

    """
    df = df.groupby(date_var).agg(count_ids=(index_var, "count")).reset_index()

    if time_step == "month":
        df["time"] = df[date_var].apply(lambda x: x[:7])
    else:
        print("Enter a good time_step such as 'month' or 'week'")

    candles = (
        df.groupby("time")
        .agg(
            low=("count_ids", "min"),
            high=("count_ids", "max"),
            mean=("count_ids", "mean"),
        )
        .reset_index()
    )
    candles["MA5"] = candles["mean"].rolling(short_trend_step).mean()
    candles["MA20"] = candles["mean"].rolling(long_trend_step).mean()

    open_date = df.groupby("time")[date_var].min().reset_index()
    open_date = pd.merge(open_date, df[[date_var, "count_ids"]], on=date_var)
    open_date = open_date[["time", "count_ids"]]
    open_date.columns = ["time", "open"]

    close_date = df.groupby("time")[date_var].max().reset_index()
    close_date = pd.merge(close_date, df[[date_var, "count_ids"]], on=date_var)
    close_date = close_date[["time", "count_ids"]]
    close_date.columns = ["time", "close"]

    candles = pd.merge(candles, open_date, on="time")
    candles = pd.merge(candles, close_date, on="time")

    """ candles["lower"] = (
        candles["mean"].rolling(context_scale).mean()
        - candles["mean"].rolling(context_scale).std() * 1.5
    )
    candles["upper"] = (
        candles["mean"].rolling(context_scale).mean()
        + candles["mean"].rolling(context_scale).std() * 1.5
    )"""

    # plot the candlesticks
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=candles.time,
                open=candles.open,
                high=candles.high,
                low=candles.low,
                close=candles.close,
                name="Semantic Candles",
            ),
            go.Scatter(
                x=candles.time,
                y=candles.MA5,
                line=dict(color="orange", width=1),
                name="Short Trend",
            ),
            go.Scatter(
                x=candles.time,
                y=candles.MA20,
                line=dict(color="green", width=1),
                name="Long Trend",
            ),
        ]
    )

    fig.update_layout(
        title=dict(
            text="<b>Semantic Candles</b>",
            font=dict(family="Arial", size=50, color="#000000"),
        ),
        height=height,
        width=width,
    )

    return fig


def average_comparison(
    df_sample,
    date_var="date",
    index_var="id",
    smoothing_scale=5,
    context_scale=20,
    height=1200,
    width=2000,
):
    """compare the difference bewteen two moving  averages"""
    df = df_sample.groupby(date_var).agg(count_ids=(index_var, "count"))

    df["trend"] = df["count_ids"].rolling(smoothing_scale).mean()
    df["long_trend"] = df["count_ids"].rolling(context_scale).mean()

    df.columns = ["y", "ma1", "ma2"]
    df = df.dropna()
    df1 = df.copy()

    # split data into chunks where averages cross each other
    df["label"] = np.where(df["ma1"] > df["ma2"], 1, 0)
    df["group"] = df["label"].ne(df["label"].shift()).cumsum()
    df = df.groupby("group")
    dfs = []
    for name, data in df:
        dfs.append(data)

    # custom function to set fill color
    def fillcol(label):
        if label >= 1:
            return "rgba(0,250,0,0.4)"
        else:
            return "rgba(250,0,0,0.4)"

    fig = go.Figure()

    for df in dfs:
        fig.add_traces(
            go.Scatter(x=df.index, y=df.ma1, line=dict(color="rgba(0,0,0,0)"))
        )

        fig.add_traces(
            go.Scatter(
                x=df.index,
                y=df.ma2,
                line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty",
                fillcolor=fillcol(df["label"].iloc[0]),
            )
        )

    # include averages
    """fig.add_traces(
        go.Scatter(x=df1.index, y=df1.ma1, line=dict(color="white", width=1))
    )

    fig.add_traces(
        go.Scatter(x=df1.index, y=df1.ma2, line=dict(color="white", width=1))
    )"""

    # include main time-series
    # fig.add_traces(go.Scatter(x=df1.index, y=df1.y, line=dict(color="black", width=2)))

    fig.update_layout(showlegend=False)
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title=dict(
            text="<b>Semantic Trend</b>",
            font=dict(size=50),
        ),
        height=height,
        width=width,
    )

    return fig


if __name__ == "__main__":
    df_sample = pd.read_csv(
        "/Volumes/OutFriend/timeline_folding/time_sample.csv", index_col=[0]
    )
    platform = "facebook"
    df_sample = df_sample[df_sample["origin"] == platform]

    """fig = trend(
        df_sample, date_var="date", index_var="id", smoothing_scale=7, context_scale=182
    )

    """

    f""""ig = semantic_candles(
        df_sample,
        date_var="date",t
        index_var="id",
        short_trend_step=5,
        long_trend_step=20,
        time_step="month",
    )

    fig_2 = trend(
        df_sample,
        date_var="date",
        index_var="id",
        smoothing_scale=7,
        context_scale=30,
        stv_scale=0.8,
    )"""

    df_sample["date"] = pd.to_datetime(df_sample["date"])
    # df_sample = df_sample[df_sample["date"].dt.year >= 2020]

    fig_3 = average_comparison(
        df_sample, date_var="date", index_var="id", smoothing_scale=7, context_scale=31
    )

    fig_3.show()
