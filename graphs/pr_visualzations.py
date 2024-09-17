import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
from wordcloud import WordCloud
from typing import Dict, List, Tuple
from plotly.subplots import make_subplots

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe by adding derived columns for various date components.
    
    Args:
        df (pd.DataFrame): Input dataframe with a 'date' column.
    
    Returns:
        pd.DataFrame: Preprocessed dataframe with additional columns.
    """
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['closed_at'] = pd.to_datetime(df['closed_at'])
    
    df['duration'] = (df['closed_at'] - df['created_at']).dt.total_seconds() / 3600
    return df

def plot_additions_deletions(df: pd.DataFrame) -> go.Figure:
    """
    Create a grouped bar plot of additions and deletions per pull request.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure(data=[
        go.Bar(name='Additions', x=df['number'], y=df['additions']),
        go.Bar(name='Deletions', x=df['number'], y=df['deletions'])
    ])
    fig.update_layout(title='Additions and Deletions per Pull Request',
                      xaxis_title='Pull Request Number',
                      yaxis_title='Number of Changes',
                      barmode='group')
    return fig

def plot_review_cycle_time(df: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot of review cycle time vs additions.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    df = preprocess_dataframe(df)
    return px.scatter(df, x='review_cycle_time', y='additions',
                      hover_data=['number', 'title', 'author'],
                      labels={'review_cycle_time': 'Review Cycle Time (days)',
                              'additions': 'Number of Additions'},
                      title='Review Cycle Time vs Additions')

def plot_pr_state_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart of pull request state distribution.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    df = preprocess_dataframe(df)
    state_counts = df['state'].value_counts()
    return px.pie(values=state_counts.values, names=state_counts.index, 
                  title='Pull Request State Distribution')

def plot_pr_creation_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Create a line plot of pull request creation timeline.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    df = preprocess_dataframe(df)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    df_grouped = df.groupby(df['created_at'].dt.date).size().reset_index(name='pr_count')
    
    return px.line(df_grouped, x='created_at', y='pr_count', 
                   title='Pull Request Creation Timeline',
                   labels={'created_at': 'Date', 'pr_count': 'Number of PRs Created'})
    
    
def plot_pr_bubble_chart(df):
    df = preprocess_dataframe(df)
    fig = px.scatter(df, x='review_cycle_time', y='duration', size='additions', color='author',
                     hover_name='title', text='number',
                     labels={'review_cycle_time': 'Review Cycle Time (days)',
                             'duration': 'PR Duration (hours)',
                             'additions': 'Additions'},
                     title='PR Complexity: Review Time vs Duration vs Size')
    fig.update_traces(textposition='top center')
    return fig

def plot_author_contribution(df):
    df = preprocess_dataframe(df)
    author_stats = df.groupby('author').agg({
        'number': 'count',
        'additions': 'sum',
        'deletions': 'sum',
        'review_cycle_time': 'mean'
    }).reset_index()
    
    fig = make_subplots(rows=2, cols=2, 
                        specs=[[{'type': 'domain'}, {'type': 'domain'}],
                               [{'type': 'domain'}, {'type': 'domain'}]],
                        subplot_titles=('PRs Count', 'Total Additions', 'Total Deletions', 'Avg Review Cycle Time'))
    
    fig.add_trace(go.Pie(labels=author_stats['author'], values=author_stats['number'], name="PRs Count"), 1, 1)
    fig.add_trace(go.Pie(labels=author_stats['author'], values=author_stats['additions'], name="Total Additions"), 1, 2)
    fig.add_trace(go.Pie(labels=author_stats['author'], values=author_stats['deletions'], name="Total Deletions"), 2, 1)
    fig.add_trace(go.Pie(labels=author_stats['author'], values=author_stats['review_cycle_time'], name="Avg Review Cycle Time"), 2, 2)
    
    fig.update_layout(title_text="Author Contribution Analysis")
    return fig

def plot_pr_efficiency(df):
    df = preprocess_dataframe(df)
    df['efficiency'] = df['additions'] / (df['duration'] + 1)  # Adding 1 to avoid division by zero
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['number'], y=df['efficiency'], name='PR Efficiency'))
    fig.add_trace(go.Scatter(x=df['number'], y=df['review_cycle_time'], mode='lines+markers', name='Review Cycle Time'))
    
    fig.update_layout(title='PR Efficiency (Additions per Hour) vs Review Cycle Time',
                      xaxis_title='PR Number',
                      yaxis_title='Efficiency / Review Cycle Time')
    return fig

def plot_timeline_heatmap(df):
    df = preprocess_dataframe(df) 
    df['month'] = df['created_at'].dt.to_period('M')
    df['day'] = df['created_at'].dt.day
    heatmap_data = df.pivot_table(values='additions', index='day', columns='month', aggfunc='sum', fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
                   z=heatmap_data.values,
                   x=heatmap_data.columns.astype(str),
                   y=heatmap_data.index,
                   colorscale='Viridis'))

    fig.update_layout(
        title='PR Activity Heatmap',
        xaxis_title='Month',
        yaxis_title='Day of Month')
    return fig