import pandas as pd
import numpy as np
import plotly.graph_objs as go
from ..specificity import specificity
from ..basic_class import BasicSemantics


def wrap_by_word(string, n_words):
    """returns a string where \\n is inserted between every n words"""
    a = string.split()
    ret = ""
    for i in range(0, len(a), n_words):
        ret += " ".join(a[i : i + n_words]) + "<br>"

    return ret


class SemanticsTrends(BasicSemantics):
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
        terms_multiprocessing=True,
        docs_multiprocessing=True,
        docs_dimension_reduction=5,
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
            terms_multiprocessing=terms_multiprocessing,
            docs_multiprocessing=docs_multiprocessing,
            docs_dimension_reduction=docs_dimension_reduction,
        )

    def fit(self, date_var):
        # Only Fit on the date
        self.date_var = date_var
        self.data["date_format"] = self.data[self.date_var]

    def moving_average_comparison(
        self,
        smoothing_scale=5,
        context_scale=20,
        height=800,
        width=800,
        top_terms_period=10,
    ):
        """compare the difference bewteen two moving  averages"""
        df = self.data.groupby("date_format").agg(count_ids=(self.index_var, "count"))

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

        self.dfs = dfs
        res_date = self.specific_terms_time(top_n=top_terms_period)
        res_date["terms"] = res_date["terms"].apply(lambda x: wrap_by_word(x, 10))

        # custom function to set fill color
        def fillcol(label):
            if label >= 1:
                return "rgba(0,250,0,0.4)"
            else:
                return "rgba(250,0,0,0.4)"

        fig = go.Figure()

        for df in dfs:
            df = pd.merge(df.reset_index(), res_date.drop("label", axis=1), on="group")
            try:
                fig.add_traces(
                    go.Scatter(
                        x=df["date_format"],
                        y=df.ma1,
                        line=dict(color="rgba(0,0,0,0)"),
                        text=df["terms"],
                    )
                )

                fig.add_traces(
                    go.Scatter(
                        x=df["date_format"],
                        y=df.ma2,
                        line=dict(color="rgba(0,0,0,0)"),
                        fill="tonexty",
                        fillcolor=fillcol(df["label"].iloc[0]),
                    )
                )
            except:
                pass

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
            self.data.groupby("date_format")
            .agg(count_ids=(self.index_var, "count"))
            .reset_index()
        )

        df["year-month"] = df["date_format"].astype(str)
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

        open_date = df.groupby("year-month")["date_format"].min().reset_index()
        open_date = pd.merge(
            open_date, df[["date_format", "count_ids"]], on="date_format"
        )
        open_date = open_date[["year-month", "count_ids"]]
        open_date.columns = ["year-month", "open"]

        close_date = df.groupby("year-month")["date_format"].max().reset_index()
        close_date = pd.merge(
            close_date, df[["date_format", "count_ids"]], on="date_format"
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

    def classic_trend(
        self,
        smoothing_scale=7,
        context_scale=182,
        height=1000,
        width=2000,
        stv_scale=1.5,
    ):

        """Count the number of documents in a specific date time"""

        df = (
            self.data.groupby("date_format")
            .agg(count_ids=(self.index_var, "count"))
            .reset_index()
        )
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
            x=df["date_format"],
            y=df["count_ids"],
            mode="lines",
            marker=dict(
                # color='rgb(0, 0, 0)',
            ),
            name="data",
        )

        trace2 = go.Scatter(
            x=df["date_format"],
            y=df["trend"],
            mode="lines",
            marker=dict(
                # color='#5E88FC',
                # symbol='circle-open'
            ),
            name="moving_average",
        )

        trace3 = go.Scatter(
            x=df["date_format"],
            y=df["lower"],
            mode="lines",
            marker=dict(
                color="grey",
                # symbol='circle-open'
            ),
            name="lower",
        )

        trace4 = go.Scatter(
            x=df["date_format"],
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
                text="<b>Temporal Trend</b>",
                font=dict(family="Arial", size=20, color="#000000"),
            ),
            height=height,
            width=width,
        )

        return fig

    def specific_terms_time(self, top_n=10):
        """
        Get the specific terms by time period between two moving average. THis is a way to understand
        what are the topics when there is an outburst of comments
        """

        df_terms = pd.merge(self.data, self.df_indexed.reset_index(), on=self.index_var)

        # Merge timeline and terms
        df_date = pd.concat([x for x in self.dfs]).reset_index()
        fin = pd.merge(df_date, df_terms, on="date_format")
        fin = fin[["date_format", "label", "group", "main form"]]

        _, _, edge = specificity(fin, X="group", Y="main form", Z=None, top_n=top_n)
        final = pd.merge(edge, df_date, on="group")

        def join(x):
            return " | ".join(set(x))

        # Group by the "group", meanign the period where the short moving average is above or
        # under the long-trend moving-average
        res_date = (
            final.groupby(["group", "label"])
            .agg(
                terms=("main form", join),
                min_date=("date_format", "min"),
                max_date=("date_format", "max"),
            )
            .reset_index()
        )

        # Get only the top terms
        # res_date = res_date[res_date["label"] == 1]
        # res_date["lenght"] = res_date["max_date"] - res_date["min_date"]

        self.df_date_specificity = res_date

        return res_date
