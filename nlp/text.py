import pandas as pd
import logging
import google.generativeai as genai
import os
from collections import Counter
from dotenv import load_dotenv
import en_core_web_sm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Check for API key
api_key = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Load spaCy model
nlp = en_core_web_sm.load()

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv("data/commits.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['message_length'] = df['message'].str.len()
    return df

def extract_entities(text: str) -> list:
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'LOC']]

def preprocess_commits(df: pd.DataFrame) -> tuple:
    df['entities'] = df['message'].apply(extract_entities)
    entity_counts = Counter([entity for entities in df['entities'] for entity in entities])
    return df, entity_counts

def query_gemini(user_query: str, data: pd.DataFrame, entity_counts: Counter) -> str:
    # Prepare prompt for Gemini API
    data_str = data.to_string(index=False)
    
    prompt = f"""
    Given the following GitHub commit data:

    {data_str}

    Additional context:
    - Entities (technologies, organizations, locations) mentioned in commit messages and their frequencies: {dict(entity_counts)}
    - Total number of commits: {len(data)}
    - Date range: from {data['date'].min()} to {data['date'].max()}
    - Number of unique authors: {data['author'].nunique()}

    Please answer the following query or provide a suitable visualization do not give any code 
    return aas the image requested if requested:
    {user_query}

    Provide your answer in a clear, concise manner.
    """
    
    try:
        response = model.generate_content(prompt)
        logging.info(f"Response received: {response.text}")
        return response.text
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"

# Example usage
def text_gemini(user_query_input):
    # Load and preprocess the data
    df = load_data("../data/commits.csv")
    df, entity_counts = preprocess_commits(df)    

    # Get response from Gemini API
    response = query_gemini(user_query_input, df, entity_counts)
    return response
