import os
from github import Github
import pandas as pd
import re
from dotenv import load_dotenv
import concurrent.futures
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# GitHub access token
github_token = os.getenv('GITHUB_TOKEN')

# Initialize GitHub client
g = Github(github_token)

rate_limit = g.get_rate_limit()
print(f"Core Rate Limit: {rate_limit.core}")

def extract_repo_info(url):
    pattern = r"github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1), match.group(2)
    raise ValueError("Invalid GitHub URL")

def process_commit(commit):
    try:
        return {
            'sha': commit.sha,
            'author': commit.author.login if commit.author else 'Unknown',
            'date': commit.commit.author.date,
            'message': commit.commit.message
        }
    except AttributeError as e:
        print(f"Error processing commit {commit.sha}: {str(e)}")
        return None

def process_pull_request(pr):
    try:
        return {
            'number': pr.number,
            'title': pr.title or 'No Title',
            'author': pr.user.login if pr.user else 'Unknown',
            'created_at': pr.created_at if pr.created_at else None,
            'closed_at': pr.closed_at if pr.closed_at else None,
            'state': pr.state,
            'review_comments': pr.review_comments,
            'additions': pr.additions,
            'deletions': pr.deletions
        }
    except AttributeError as e:
        print(f"Error processing PR {pr.number}: {str(e)}")
        return None

def process_issue(issue):
    try:
        events = list(issue.get_events())
        reopen_count = sum(1 for event in events if event.event == 'reopened')
        return {
            'number': issue.number,
            'title': issue.title or 'No Title',
            'created_at': issue.created_at,
            'closed_at': issue.closed_at,
            'reopen_count': reopen_count
        }
    except AttributeError as e:
        print(f"Error processing issue {issue.number}: {str(e)}")
        return None

def collect_repo_data(owner, repo_name):
    try:
        repo = g.get_repo(f"{owner}/{repo_name}")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            commits_future = executor.submit(list, repo.get_commits())
            prs_future = executor.submit(list, repo.get_pulls(state='all', sort='created', direction='desc'))
            issues_future = executor.submit(list, repo.get_issues(state='all'))
            
            commits = commits_future.result()
            pull_requests = prs_future.result()
            issues = issues_future.result()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            commit_data = list(filter(None, executor.map(process_commit, commits)))
            pr_data = list(filter(None, executor.map(process_pull_request, pull_requests)))
            issue_data = list(filter(None, executor.map(process_issue, issues)))
        
        commits_df = pd.DataFrame(commit_data)
        prs_df = pd.DataFrame(pr_data)
        issues_df = pd.DataFrame(issue_data)
        
        metrics = calculate_metrics(commits_df, prs_df, issues_df)
        
        return {
            'commits': commits_df,
            'pull_requests': prs_df,
            'issues': issues_df,
            'metrics': metrics
        }
    except Exception as e:
        print(f"Error collecting data for repo {owner}/{repo_name}: {str(e)}")
        return None

def calculate_metrics(commits_df, prs_df, issues_df):
    metrics = {}
    
    if not commits_df.empty:
        commits_df['date'] = pd.to_datetime(commits_df['date'], errors='coerce')
        commit_velocity = commits_df.set_index('date').resample('W').size()
        metrics['average_commits_per_week'] = commit_velocity.mean()
        
        contributors_per_month = commits_df.resample('ME', on='date')['author'].nunique()
        metrics['average_active_contributors_per_month'] = contributors_per_month.mean()
    else:
        metrics['average_commits_per_week'] = 0
        metrics['average_active_contributors_per_month'] = 0
    
    if not prs_df.empty:
        prs_df['created_at'] = pd.to_datetime(prs_df['created_at'], errors='coerce')
        prs_df['closed_at'] = pd.to_datetime(prs_df['closed_at'], errors='coerce')
        prs_df['review_cycle_time'] = (prs_df['closed_at'] - prs_df['created_at']).dt.days
        metrics['average_pr_review_cycle_time'] = prs_df['review_cycle_time'].mean()
        metrics['average_pr_size'] = prs_df['additions'].add(prs_df['deletions']).mean()
        metrics['average_pr_review_time'] = prs_df['review_comments'].mean()
    else:
        metrics['average_pr_review_cycle_time'] = 0
        metrics['average_pr_size'] = 0
        metrics['average_pr_review_time'] = 0

    if not issues_df.empty:
        metrics['issue_reopening_rate'] = issues_df['reopen_count'].mean()
        bug_issues = issues_df[issues_df['title'].str.contains('bug', case=False, na=False)]
        feature_issues = issues_df[issues_df['title'].str.contains('feature', case=False, na=False)]
        metrics['bug_to_feature_ratio'] = len(bug_issues) / len(feature_issues) if len(feature_issues) > 0 else 0
    else:
        metrics['issue_reopening_rate'] = 0
        metrics['bug_to_feature_ratio'] = 0
    
    if not commits_df.empty:
        top_contributors = commits_df['author'].value_counts().head(10)
        metrics['top_contributors_by_commits'] = top_contributors.to_dict()
    else:
        metrics['top_contributors_by_commits'] = {}
    
    return metrics

def main():
    repo_url = input("Enter the GitHub repository URL: ")
    try:
        owner, repo_name = extract_repo_info(repo_url)
        repo_data = collect_repo_data(owner, repo_name)
        
        if repo_data:
            print("\nRepository Metrics:")
            for key, value in repo_data['metrics'].items():
                print(f"{key}: {value}")
            
            # Save data to CSV files
            repo_data['commits'].to_csv(f"{repo_name}_commits.csv", index=False)
            repo_data['pull_requests'].to_csv(f"{repo_name}_pull_requests.csv", index=False)
            repo_data['issues'].to_csv(f"{repo_name}_issues.csv", index=False)
            
            print(f"\nData saved to CSV files: {repo_name}_commits.csv, {repo_name}_pull_requests.csv, {repo_name}_issues.csv")
        else:
            print("Failed to collect repository data.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")