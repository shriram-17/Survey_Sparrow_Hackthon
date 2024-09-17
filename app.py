import os
import streamlit as st
import plotly.graph_objects as go
import plotly
from github_collection import extract_repo_info, collect_repo_data
from graphs.commits_visualization import (
    plot_commit_timeline,
    plot_commits_by_author,
    plot_commit_types_pie,
    plot_commit_activity_gantt,
    plot_commit_activity_heatmap,
    plot_commit_frequency,
    plot_commit_day_distribution,
    plot_commit_hour_distribution,
    plot_commit_message_wordcloud,
    plot_interactive_commit_network
)

from graphs.pr_visualzations import (
    plot_additions_deletions,
    plot_review_cycle_time,
    plot_pr_state_distribution,
    plot_pr_creation_timeline,
    plot_pr_bubble_chart,
    plot_author_contribution,
    plot_timeline_heatmap,
    plot_pr_efficiency
)
from PIL import Image

from nlp.text import text_gemini
from nlp.graph import image_gemini

# Set page configuration with a theme
st.set_page_config(
    page_title="GitHub Repository Analytics Dashboard",
    page_icon=":octocat:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define custom CSS styles
CUSTOM_CSS = """
    /* Navbar styling */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #007bff;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .navbar h1 {
        margin: 0;
    }
    .navbar a {
        color: white;
        text-decoration: none;
        margin-left: 20px;
        font-weight: bold;
    }
    .navbar a:hover {
        text-decoration: underline;
    }
    /* Sidebar width and content */
    .css-1lcbmhc.e1fqkh3o4 {
        width: 220px;
    }
    .css-1v3fvcr {
        width: 220px;
    }
    /* Main content styling */
    .main {
        background-color: #f8f9fa;
        color: #343a40;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #007bff;
        font-family: 'Helvetica', sans-serif;
    }
    /* Button styling */
    .stButton button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: white;
    }
    .sidebar .sidebar-content a {
        color: #ced4da;
    }
    .sidebar .sidebar-content a:hover {
        color: #007bff;
    }
    /* Card styling */
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metrics-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 20px;
        margin-top: 30px;
    }
    .metric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        flex: 1 1 calc(33.33% - 20px);
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric h3 {
        color: #6c757d;
        font-size: 1.2em;
        margin-bottom: 10px;
    }
    .metric h1 {
        color: #007BFF;
        font-size: 2.5em;
        margin: 0;
    }
    .metric:hover {
        transform: scale(1.05);
    }
    @media (max-width: 768px) {
        .metric {
            flex: 1 1 100%;
        }
    }
    .graph-container {
        margin-bottom: 30px;
    }
    .graph-container .stPlotlyChart {
        height: 400px;
    }
"""

# Cache the data fetching to avoid re-fetching it multiple times
@st.cache_data
def get_repo_data(owner, repo_name):
    return collect_repo_data(owner, repo_name)

# Initialize session_state data if it doesn't exist
if 'data' not in st.session_state:
    st.session_state['data'] = None
def main():
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

    st.sidebar.title(":bar_chart: GitHub Repository Analytics")
    option = st.sidebar.radio("Navigate", ["Home", "Commits", "Pull Requests", "Developer Activity", "NLP Module - Text Only Queries","NLP Module - Visualization Only Queries"])

    if option == "Home":
        home_page()
    elif option == "Commits":
        commits_page()
    elif option == "Pull Requests":
        pull_requests_page()
    elif option == "Developer Activity":
        developer_page()
    elif option == "NLP Module - Text Only Queries":
        nlp_module_text_page()
    elif option == "NLP Module - Visualization Only Queries":
        nlp_module_image_page()

def home_page():
    st.title("GitHub Repository Analytics Dashboard")
    st.write("Enter a GitHub repository URL to fetch commit and pull request data.")

    repo_url = st.text_input("GitHub Repository URL:", "https://github.com/aws-samples/aws-copilot-sample-service")

    if st.button("Fetch Data"):
        try:
            owner, repo_name = extract_repo_info(repo_url)
            data = get_repo_data(owner, repo_name)
            st.session_state['data'] = data

            st.title(f"Analytics Dashboard for {owner}/{repo_name}")
            display_metrics(data)
        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

def commits_page():
    try:
        data = st.session_state['data']
        if data and not data['commits'].empty:
            st.subheader("Commit Data")
            st.dataframe(data['commits'])
            os.makedirs('data', exist_ok=True)
            data['commits'].to_csv('data/commits.csv', index=False)

            st.subheader("Commit Visualizations")
            display_commit_visualizations(data['commits'])
        else:
            st.warning("No commit history found.")
    except (NameError, KeyError):
        st.warning("No data fetched yet. Please enter a repository URL on the Home page.")

def pull_requests_page():
    try:
        data = st.session_state['data']
        if data and not data['pull_requests'].empty:
            st.subheader("Pull Request Data")
            st.dataframe(data['pull_requests'])
            os.makedirs('data', exist_ok=True)
            data['pull_requests'].to_csv('data/pull_requests.csv', index=False)

            st.subheader("Pull Request Visualizations")
            display_pull_request_visualizations(data['pull_requests'])
        else:
            st.warning("No pull request history found.")
    except (NameError, KeyError):
        st.warning("No data fetched yet. Please enter a repository URL on the Home page.")

def nlp_module_text_page():
    st.title("NLP Module for Text ")
    
    user_query_input = st.text_input("Ask a question about the commit data:")
    
    if st.button("Analyze"):
        if user_query_input:   
            response = text_gemini(user_query_input)
            st.write(response)
        else:
            st.warning("Please enter a query.")
def nlp_module_image_page():
    st.title("NLP Module For Image ")
    
    user_query_input = st.text_input("Ask a question about the commit data:")
    if st.button("Analyze"):
        image = image_gemini(user_query_input)
        if image:
            st.plotly_chart(image)
        else:
            st.error("The generated image file 'generated_plot.png' was not found.")
            
def settings_page():
    st.title("Settings")
    st.write("This section will contain settings and configurations.")

def display_metrics(data):
    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><h3>Total Commits</h3><h1>{len(data['commits'])}</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><h3>Total Pull Requests</h3><h1>{len(data['pull_requests'])}</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><h3>Active Authors</h3><h1>{data['commits']['author'].nunique()}</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><h3>Average Commits Per Week</h3><h1>{data['metrics']['average_commits_per_week']:.2f}</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><h3>Average PR Review Time (hrs)</h3><h1>{data['metrics']['average_pr_review_time']:.2f}</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><h3>Bug to Feature Ratio</h3><h1>{data['metrics']['bug_to_feature_ratio']:.2f}</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><h3>Average Active Contributors/Month</h3><h1>{data['metrics']['average_active_contributors_per_month']:.2f}</h1></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'><h3>Issue Reopening Rate</h3><h1>{data['metrics']['issue_reopening_rate']:.2f}</h1></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def display_commit_visualizations(commits):
    try:
        st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
        
        plotting_functions = [
            plot_commit_timeline,
            plot_commits_by_author,
            plot_commit_types_pie,
            plot_commit_day_distribution,
            plot_commit_hour_distribution,
            plot_interactive_commit_network,
            plot_commit_message_wordcloud
        ]
        
        for plot_func in plotting_functions:
            try:
                fig = plot_func(commits)
                if isinstance(fig, (dict, list)) or isinstance(fig, plotly.graph_objs.Figure):
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"{plot_func.__name__} did not return a valid figure.")
            except Exception as e:
                st.error(f"Error generating {plot_func.__name__}: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating visualizations: {e}")
        
def display_pull_request_visualizations(pull_requests):
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.plotly_chart(plot_review_cycle_time(pull_requests), use_container_width=True)
    st.plotly_chart(plot_pr_state_distribution(pull_requests), use_container_width=True)
    st.plotly_chart(plot_pr_creation_timeline(pull_requests), use_container_width=True)
    st.plotly_chart(plot_pr_bubble_chart(pull_requests), use_container_width=True)
    st.plotly_chart(plot_author_contribution(pull_requests), use_container_width=True)
    st.plotly_chart(plot_timeline_heatmap(pull_requests), use_container_width=True)
    st.plotly_chart(plot_pr_efficiency(pull_requests), use_container_width=True)
    st.plotly_chart(plot_additions_deletions(pull_requests), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Developer Activity Page
def developer_page():
    try:
        data = st.session_state['data']
        if data and not data['commits'].empty:
            st.subheader("Developer-Specific Data")

            # Extract unique authors from the commits data
            authors = data['commits']['author'].unique()
            selected_author = st.selectbox("Select a Developer:", authors)

            # Filter the commit data for the selected author
            developer_commits = data['commits'][data['commits']['author'] == selected_author]

            # Display basic metrics for the selected developer
            st.markdown(f"### Activity for {selected_author}")
            st.markdown(f"**Total Commits:** {len(developer_commits)}")
            st.markdown(f"**First Commit Date:** {developer_commits['date'].min()}")
            st.markdown(f"**Last Commit Date:** {developer_commits['date'].max()}")

            st.subheader("Developer Commit Visualizations")

            st.plotly_chart(plot_commit_timeline(developer_commits), use_container_width=True)
            st.plotly_chart(plot_commit_types_pie(developer_commits), use_container_width=True)
            st.plotly_chart(plot_commit_day_distribution(developer_commits), use_container_width=True)
    
        else:
            st.warning("No commit history found.")
    except (NameError, KeyError):
        st.warning("Error here Something")
        st.warning("No data fetched yet. Please enter a repository URL on the Home page.")

if __name__ == "__main__":
    main()
