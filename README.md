Here's the updated README file with the new link for the summary report:

---

# GitHub Repository Analytics Tool

## Overview
This project provides an interactive dashboard for analyzing GitHub repository data, including commits, pull requests, and developer activities. It utilizes Streamlit for the user interface, the GitHub API for data collection, and Plotly for data visualization. Additionally, Natural Language Processing (NLP) capabilities are integrated using Google's Gemini AI for generating AI-driven insights based on user queries.

## Table of Contents
1. [Main Application](#main-application)
2. [GitHub Data Collection](#github-data-collection)
3. [NLP and Visualization Module](#nlp-and-visualization-module)
4. [Pull Request Visualizations](#pull-request-visualizations)
5. [Commit Visualizations](#commit-visualizations)
6. [Installation and Execution](#installation-and-execution)
7. [Summary Report](#summary-report)

---

## Main Application <a name="main-application"></a>

### Overview
This file contains the main Streamlit application that provides a user-friendly interface for GitHub repository analysis. It allows users to navigate through various pages to view commit, pull request, and developer activity data, as well as interact with NLP modules for querying the data.

### Key Components
- **Streamlit Setup**: Initializes the Streamlit app with custom CSS for styling.
- **Data Fetching**: Retrieves GitHub repository data using caching to optimize performance.
- **Navigation**: A sidebar allows users to navigate between Home, Commits, Pull Requests, Developer Activity, and NLP Modules.
- **Visualizations**: Displays various charts using Plotly for analyzing commits and pull requests.
- **NLP Integration**: Includes text-based and image-based NLP modules for querying repository data.

### Main Functions
- `main()`: The entry point of the application that handles navigation between pages.
- `home_page()`: Displays the main dashboard and fetches repository data.
- `commits_page()`: Displays commit data and related visualizations.
- `pull_requests_page()`: Displays pull request data and visualizations.
- `developer_page()`: Provides developer-specific analytics.
- `nlp_module_text_page()`: Processes text-based NLP queries.
- `nlp_module_image_page()`: Processes image-based NLP queries.

### Usage
Run this file to start the Streamlit application and use the web interface for repository analytics.

---

## GitHub Data Collection <a name="github-data-collection"></a>

### Overview
This script fetches data from a GitHub repository using the GitHub API and processes it to generate useful analytics.

### Key Components
- **GitHub API Integration**: Uses the PyGithub library to interact with GitHub's API.
- **Concurrent Processing**: Utilizes concurrent processing to handle commits, pull requests, and issues in parallel.
- **Data Processing**: Converts raw data from GitHub into pandas DataFrames for easy analysis.
- **Metrics Calculation**: Computes metrics like commit frequency, pull request review times, and more.

### Main Functions
- `extract_repo_info(url)`: Extracts owner and repository name from a GitHub URL.
- `process_commit(commit)`, `process_pull_request(pr)`, `process_issue(issue)`: Processes individual items from the GitHub API.
- `collect_repo_data(owner, repo_name)`: Main function that collects all repository data.
- `calculate_metrics(commits_df, prs_df, issues_df)`: Computes various repository metrics.

### Usage
This script can either be run standalone to collect repository data or imported into other scripts to integrate GitHub data fetching.

---

## NLP and Visualization Module <a name="nlp-and-visualization-module"></a>

### Overview
This module integrates Natural Language Processing (NLP) capabilities and visualizations for analyzing commit data from a GitHub repository. It uses Google's Gemini AI for answering queries and generating interactive visualizations.

### Key Components
- **Data Loading**: Loads commit data from a CSV file.
- **NLP Processing**: Uses spaCy for entity extraction from commit messages.
- **Gemini AI Integration**: Queries Google's Gemini AI for responses based on user input.
- **Visualization**: Generates interactive charts and graphs using Plotly, based on the AI-generated insights.

### Main Functions
- `load_data()`: Loads and preprocesses commit data.
- `extract_entities(text)`: Extracts named entities using spaCy.
- `preprocess_commits(df)`: Preprocesses commit data for analysis.
- `query_gemini(user_query, data, entity_counts)`: Sends user queries to Gemini AI.
- `image_gemini(user_query)`: Generates visualizations based on the AI's responses.

### Usage
This module is used within the main application to provide advanced NLP insights and visualizations for GitHub repository data.

---

## Pull Request Visualizations <a name="pull-request-visualizations"></a>

### Overview
This module provides functions for visualizing pull request data using Plotly.

### Key Components
- Preprocessing of pull request data.
- Several visualization functions, including bar charts, pie charts, scatter plots, and heatmaps.

### Main Functions
- `preprocess_dataframe(df)`: Preprocesses the pull request DataFrame.
- `plot_additions_deletions(df)`: Creates a bar plot for additions and deletions.
- `plot_review_cycle_time(df)`: Generates a scatter plot for review cycle times.
- `plot_pr_state_distribution(df)`: Creates a pie chart of pull request states.
- `plot_pr_bubble_chart(df)`: Creates a bubble chart for pull request complexity.

### Usage
These visualizations can be integrated with the main application to display pull request analytics.

---

## Commit Visualizations <a name="commit-visualizations"></a>

### Overview
This module contains functions for visualizing commit data, including timelines, heatmaps, and word clouds.

### Key Components
- Preprocessing of commit data.
- Visualization functions, including line plots, pie charts, and network diagrams.

### Main Functions
- `preprocess_dataframe(df)`: Preprocesses the commit DataFrame.
- `plot_commit_timeline(df)`: Creates a timeline for commits.
- `plot_commit_frequency(df)`: Creates bar plots of commit frequency by hour, week, and month.
- `plot_commit_message_wordcloud(df)`: Generates a word cloud of commit messages.

### Usage
These visualizations can be integrated with the main application to display commit analytics.

---

## Installation and Execution <a name="installation-and-execution"></a>

### Prerequisites
Ensure that you have Python installed on your system. You will also need to have the following dependencies installed, which are specified in the `requirements.txt` file.

### Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/your-username/github-repo-analytics.git
    cd github-repo-analytics
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/MacOS
    venv\Scripts\activate  # For Windows
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To run the Streamlit application, execute the following command:

```bash
streamlit run app.py
```

This will start a local web server, and you can access the app by navigating to `http://localhost:8501` in your web browser.

### Requirements
The `requirements.txt` file should include the following dependencies (adjust according to your actual needs):

```
streamlit
pandas
plotly
PyGithub
spacy
google-cloud
gemini-ai
```

### Notes
- Make sure to set up any necessary API keys for GitHub and Gemini AI in environment variables.
- The application uses caching to reduce the number of API calls to GitHub, so data will be stored locally for future use.

### Troubleshooting
If you encounter any issues while running the app, ensure that all dependencies are correctly installed and that your API keys are correctly set up.

---

## Summary Report <a name="summary-report"></a>

For a comprehensive summary report of the GitHub Repository Analytics Tool, including methodology, findings, and recommendations, please refer to the [Summary Report](https://docs.google.com/document/d/1OTQQY6wwZToImEzKnmw6fA9791hZJOrv92s873ThbA8/edit).

---

## Conclusion
This GitHub Repository Analytics Tool provides a comprehensive interface for analyzing repository data, generating visualizations, and using NLP queries to gain insights. Through its integration of GitHub API, Plotly, and Google's Gemini AI, the tool offers a powerful and interactive way to monitor repository activity and developer contributions.

---
