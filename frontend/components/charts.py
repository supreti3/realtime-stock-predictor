from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.line(df, x=x, y=y, template="plotly_dark", title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def multi_line_chart(df: pd.DataFrame, x: str, ys: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for col in ys:
        fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=col))
    fig.update_layout(template="plotly_dark", title=title, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def allocation_pie(df: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
    fig = px.pie(df, names=names, values=values, template="plotly_dark", title=title, hole=0.35)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def heatmap(corr_df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale="Viridis", title=title)
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10))
    return fig

