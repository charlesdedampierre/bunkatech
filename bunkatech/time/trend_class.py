from ..basic_class import BasicSemantics
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.io as pio
import plotly.express as px


class SemanticsTrend(BasicSemantics):
    def __init__(self, data, text_var, index_var, date_var) -> None:
        super().__init__()
        self.data = data
        self.text_var = text_var
        self.index_var = index_var
        self.date_var = date_var
        self.data[self.date_var] = pd.to_datetime(self.data[self.date_var])

    def fit(
        self,
        extract_terms=True,
        docs_embedding=True,
        terms_embedding=True,
        sample_size_terms=500,
        terms_limit=500,
        terms_ents=True,
        terms_ngrams=(1, 2),
        terms_ncs=True,
        terms_include_pos=["NOUN", "PROPN", "ADJ"],
        terms_include_types=["PERSON", "ORG"],
        terms_embedding_model="distiluse-base-multilingual-cased-v1",
        docs_embedding_model="distiluse-base-multilingual-cased-v1",
        language="en",
    ):

        super().fit(data=self.data, text_var=self.text_var, index_var=self.index_var)

        if extract_terms:
            super().extract_terms(
                sample_size=sample_size_terms,
                limit=terms_limit,
                ents=terms_ents,
                ncs=terms_ncs,
                ngrams=terms_ngrams,
                include_pos=terms_include_pos,
                include_types=terms_include_types,
                language=language,
            )

        if terms_embedding:
            super().terms_embeddings(embedding_model=terms_embedding_model)

        if docs_embedding:
            super().embeddings(embedding_model=docs_embedding_model)

    def moving_average_comparison(
        self,
        smoothing_scale=5,
        context_scale=20,
        height=1200,
        width=2000,
    ):
        """compare the difference bewteen two moving  averages"""
        df = self.data.groupby(self.date_var).agg(count_ids=(self.index_var, "count"))

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
                font=dict(size=10),
            ),
            height=height,
            width=width,
        )

        return fig

    def semantic_candles(
        self,
        short_trend_step=5,
        long_trend_step=20,
        height=1000,
        width=2300,
    ):
        """Display the semantics as candle
        insert AAAA-MM-DD. The algorithmn groups by months and take as a year scale

        """
        df = (
            self.data.groupby(self.date_var)
            .agg(count_ids=(self.index_var, "count"))
            .reset_index()
        )

        df["year-month"] = df[self.date_var].astype(str)
        df["year-month"] = df["year-month"].apply(lambda x: x[:7])
        candles = (
            df.groupby("year-month")
            .agg(
                low=("count_ids", "min"),
                high=("count_ids", "max"),
                mean=("count_ids", "mean"),
            )
            .reset_index()
        )
        candles["MA5"] = candles["mean"].rolling(short_trend_step).mean()
        candles["MA20"] = candles["mean"].rolling(long_trend_step).mean()

        open_date = df.groupby("year-month")[self.date_var].min().reset_index()
        open_date = pd.merge(
            open_date, df[[self.date_var, "count_ids"]], on=self.date_var
        )
        open_date = open_date[["year-month", "count_ids"]]
        open_date.columns = ["year-month", "open"]

        close_date = df.groupby("year-month")[self.date_var].max().reset_index()
        close_date = pd.merge(
            close_date, df[[self.date_var, "count_ids"]], on=self.date_var
        )
        close_date = close_date[["year-month", "count_ids"]]
        close_date.columns = ["year-month", "close"]

        candles = pd.merge(candles, open_date, on="year-month")
        candles = pd.merge(candles, close_date, on="year-month")

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
                    x=candles["year-month"],
                    open=candles.open,
                    high=candles.high,
                    low=candles.low,
                    close=candles.close,
                    name="Semantic Candles",
                ),
                go.Scatter(
                    x=candles["year-month"],
                    y=candles.MA5,
                    line=dict(color="orange", width=1),
                    name="Short Trend",
                ),
                go.Scatter(
                    x=candles["year-month"],
                    y=candles.MA20,
                    line=dict(color="green", width=1),
                    name="Long Trend",
                ),
            ]
        )

        fig.update_layout(
            title=dict(
                text="<b>Semantic Candles</b>",
                font=dict(family="Arial", size=20, color="#000000"),
            ),
            height=height,
            width=width,
        )

        self.candles = candles

        return fig
