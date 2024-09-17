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
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    df['hour'] = df['date'].dt.hour
    df['week'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['day_of_week'] = df['date'].dt.day_name()
    return df

def plot_commit_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Create a line plot of commit timeline.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    df = preprocess_dataframe(df)
    df_grouped = df.groupby('date_only').size().reset_index(name='commit_count')
    return px.line(df_grouped, x='date_only', y='commit_count', title='Commit Timeline',
                   labels={'date_only': 'Date', 'commit_count': 'Number of Commits'})

def plot_commits_by_author(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar plot of commits by author.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    df = preprocess_dataframe(df)
    author_counts = df['author'].value_counts().reset_index(name='number_of_commits')
    
    return px.bar(author_counts, x='author', y='number_of_commits', color='author', title='Commits by Author',
                  labels={'author': 'Author', 'number_of_commits': 'Number of Commits'})

def plot_commit_types_pie(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart of commit types.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        go.Figure: Plotly figure object.
    """
 
    commit_types = df['message'].str.extract(r'^(\w+):', expand=False).fillna('Other')
    
   
    commit_types_counts = commit_types.value_counts().reset_index(name='count')
    commit_types_counts.columns = ['commit_type', 'count']  # Rename columns for clarity

    return px.pie(commit_types_counts, values='count', names='commit_type', title='Commit Types')


def create_commit_network(df: pd.DataFrame) -> nx.DiGraph:
    """
    Create a directed graph representing the commit network.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        nx.DiGraph: NetworkX directed graph object.
    """
    G = nx.DiGraph()
    
    for i, (_, row) in enumerate(df.iterrows()):
        G.add_node(i, author=row['author'], message=row['message'], date=row['date'])
        if i > 0:
            G.add_edge(i-1, i)
    
    return G

def plot_interactive_commit_network(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive network plot of commits.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    G = create_commit_network(df)
    pos = nx.spring_layout(G, seed=42)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepare node traces
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [f"Commit {node}<br>Author: {G.nodes[node]['author']}<br>Date: {G.nodes[node]['date']}<br>Message: {G.nodes[node]['message']}" for node in G.nodes()]
    node_adjacencies = [len(list(G.neighbors(node))) for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y,
        text=node_text,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=node_adjacencies,
            size=10,
            colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'),
            line_width=2
        )
    )
    
    # Create subplots: main plot for network, secondary plot for commit details
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], 
                        specs=[[{"type": "scatter"}, {"type": "table"}]])
    
    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    
    # Add table for commit details
    fig.add_trace(
        go.Table(
            header=dict(values=["Commit ID", "Author", "Date", "Message"],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[
                [i for i in G.nodes()],
                [G.nodes[i]['author'] for i in G.nodes()],
                [G.nodes[i]['date'] for i in G.nodes()],
                [G.nodes[i]['message'] for i in G.nodes()]
            ],
            fill_color='lavender',
            align='left')
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Interactive Commit Network Diagram',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def plot_commit_activity_gantt(df: pd.DataFrame) -> go.Figure:
    """
    Create a Gantt chart of commit activity.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    return px.timeline(df, x_start="date", x_end="date", y="author", color="author", title='Commit Activity Gantt Chart',
                       labels={'date': 'Date', 'author': 'Author'})

def plot_commit_activity_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap of commit activity.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    commit_counts = df.groupby(['date_only', 'author']).size().unstack(fill_value=0)
    
    return go.Figure(data=go.Heatmap(z=commit_counts.values, x=commit_counts.columns, y=commit_counts.index, colorscale='Viridis'),
                     layout=go.Layout(title='Commit Activity Heatmap', xaxis_title='Author', yaxis_title='Date'))

def plot_commit_frequency(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """
    Create bar plots of commit frequency by hour, week, and month.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe.
    
    Returns:
        Tuple[go.Figure, go.Figure, go.Figure]: Tuple of three Plotly figure objects.
    """
    df_hourly = df.groupby('hour').size().reset_index(name='commit_count')
    df_weekly = df.groupby('week').size().reset_index(name='commit_count')
    df_monthly = df.groupby('month').size().reset_index(name='commit_count')
    
    fig_hourly = px.bar(df_hourly, x='hour', y='commit_count', title='Commit Frequency by Hour of the Day')
    fig_weekly = px.bar(df_weekly, x='week', y='commit_count', title='Commit Frequency by Week')
    fig_monthly = px.bar(df_monthly, x='month', y='commit_count', title='Commit Frequency by Month')

    return fig_hourly, fig_weekly, fig_monthly

def plot_commit_day_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar plot of commit distribution by day of the week.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    df = preprocess_dataframe(df)
    day_counts = df['day_of_week'].value_counts().reset_index(name='commit_count')
    
    return px.bar(day_counts, x='day_of_week', y='commit_count', title='Commit Distribution by Day of the Week')

def plot_commit_hour_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar plot of commit distribution by hour of the day.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    df = preprocess_dataframe(df)
    hour_counts = df['hour'].value_counts().reset_index(name='commit_count')
    
    return px.bar(hour_counts, x='hour', y='commit_count', title='Commit Distribution by Hour of the Day')

def plot_commit_message_wordcloud(df: pd.DataFrame) -> go.Figure:
    """
    Create a word cloud of commit messages.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    text = ' '.join(df['message'].dropna())
    wordcloud = WordCloud(width=1000, height=1000, background_color='white').generate(text)
    
    wordcloud_image = wordcloud.to_array()
    
    fig = go.Figure(data=[go.Image(z=wordcloud_image)])
    fig.update_layout(title='Commit Message Word Cloud')
    
    return fig

def plot_commit_activity_by_branch(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar plot of commit activity by branch.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    branch_counts = df['branch'].value_counts().reset_index(name='commit_count')
    
    return px.bar(branch_counts, x='branch', y='commit_count', title='Commit Activity by Repository Branch')

