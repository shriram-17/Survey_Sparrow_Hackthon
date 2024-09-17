import os
import re
import logging
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import spacy
import en_core_web_sm

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Load spaCy model
nlp = en_core_web_sm.load()

def load_data():
    """
    Loads and preprocesses the commit data from a CSV file.
    """
    df = pd.read_csv("data/commits.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['message_length'] = df['message'].str.len()
    return df

def extract_entities(text):
    """
    Extracts named entities from text using spaCy.
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'LOC']]

def preprocess_commits(df):
    """
    Preprocesses commit messages to extract key information using NER.
    """
    # Extract entities from commit messages
    df['entities'] = df['message'].apply(extract_entities)
    
    # Count occurrences of each entity
    entity_counts = Counter([entity for entities in df['entities'] for entity in entities])
    
    # Extract file changes if available
    df['files_changed'] = df['message'].str.extract(r'Files changed:\s*(.*)')
    
    return df, entity_counts

def query_gemini(user_query: str, data: pd.DataFrame, entity_counts: Counter) -> str:
    """
    Queries the Gemini API with a user query and provided data,
    then processes and returns the path to the generated image.
    """
    # Convert DataFrame to a string representation
    data_str = data.head().to_string(index=False)
    
    # Prepare the prompt with a specific request to return executable code and save the visualizations
    prompt = f"""
    Given the following sample of GitHub commit data (first few rows) from the file 'data/commits.csv':\n
    {data_str}\n
    Additional context:
    - Entities (technologies, organizations, locations) mentioned in commit messages and their frequencies: {dict(entity_counts)}
    - Total number of commits: {len(data)}
    - Date range: from {data['date'].min()} to {data['date'].max()}
    - Number of unique authors: {data['author'].nunique()}
    
    Please answer the following query and provide Python code using only Plotly to generate interactive visualizations based on the commit data.
    Ensure the code adheres to these guidelines:
    1. The code should be complete and executable as is.
    2. Use only Plotly for visualizations.
    3. The code should include error handling where necessary.
    4. Provide comments in the code to explain the functionality of each section.
    5. Use the 'data' DataFrame that is already loaded and available in the context.
    6. Do not attempt to read the CSV file again.

    Here is the query: {user_query}
    """
    
    try:
        # Send the request to Gemini API
        response = model.generate_content(prompt)
        
        # Log the response for debugging purposes
        logging.info(f"Response received: {response.text}")
        
        # Clean and format the generated code
        cleaned_code = response.text
        cleaned_code = re.sub(r'```python', '', cleaned_code)
        cleaned_code = re.sub(r'```', '', cleaned_code)
        cleaned_code = re.sub(r'^\s*[\r\n]+', '', cleaned_code, flags=re.MULTILINE)  # Remove leading newlines
        cleaned_code = re.sub(r'\s*$', '', cleaned_code)  # Remove trailing whitespace
        
        # Create a local context with necessary variables and modules
        local_context = {
            'pd': pd,
            'plt': plt,
            'np': np,
            'data': data,
            'entity_counts': entity_counts
        }
        
        # Execute the cleaned code in the local context
        exec(cleaned_code, local_context)
        
        # Return path to generated image if it exists
        if 'fig' in local_context:
            return local_context['fig']
        
        
    except SyntaxError as e:
        logging.error(f"Syntax error in generated code: {str(e)}")
        return f"Syntax error: {str(e)}"
    
    except FileNotFoundError as e:
        logging.error(str(e))
        return str(e)
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"


def image_gemini(user_query):
    # Load and preprocess data outside of Streamlit context for testing or other use cases.
    df = load_data()
    df, entity_counts = preprocess_commits(df)

    image = query_gemini(user_query, df, entity_counts)

    return image