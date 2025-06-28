"""
Configuration settings for the Semantic Scholar paper search implementation.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Semantic Scholar API settings
API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEARCH_ENDPOINT = f"{BASE_URL}/paper/search"
BULK_SEARCH_ENDPOINT = f"{BASE_URL}/paper/search/bulk"
PAPER_DETAILS_ENDPOINT = f"{BASE_URL}/paper"

# API settings
RESULTS_PER_PAGE = 100
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
REQUEST_TIMEOUT = 30  # seconds

# Rate limiting
REQUESTS_PER_MINUTE = 100 if not API_KEY else 1000  # Increased limit with API key
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # seconds

# Fields to fetch for each paper
PAPER_FIELDS = [
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "publicationDate",
    "fieldsOfStudy",
    "citations",
    "references"
]

# Output settings
DEFAULT_OUTPUT_FILE = "research_papers.json" 