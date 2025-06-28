"""
Main module for searching papers using Semantic Scholar API.
"""

import time
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm
import yake
from keybert import KeyBERT
import fitz  # PyMuPDF
import re
import os

from config import (
    SEARCH_ENDPOINT, BULK_SEARCH_ENDPOINT, PAPER_DETAILS_ENDPOINT, PAPER_FIELDS,
    RESULTS_PER_PAGE, MAX_RETRIES, RETRY_DELAY,
    REQUEST_TIMEOUT, DELAY_BETWEEN_REQUESTS, DEFAULT_OUTPUT_FILE,
    API_KEY
)

class SemanticScholarAPI:
    def __init__(self):
        self.session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        if API_KEY:
            headers["x-api-key"] = API_KEY
        self.session.headers.update(headers)
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make an API request with retry logic and rate limiting."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                time.sleep(DELAY_BETWEEN_REQUESTS)  # Rate limiting
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise Exception(f"API request failed after {MAX_RETRIES} attempts: {str(e)}")
                time.sleep(RETRY_DELAY)
    
    def search_papers(self, keywords: List[str], progress_callback=None) -> List[Dict[str, Any]]:
        """
        Search for all available papers based on keywords using the bulk search endpoint.
        
        Args:
            keywords: List of search keywords
            progress_callback: Optional callback function to update progress
            
        Returns:
            List of paper dictionaries with abstracts
        """
        query = ' '.join(keywords)
        papers = []
        total_retrieved = 0
        token = None
        
        while True:
            params = {
                "query": query,
                "fields": "title,url,authors,year,abstract,fieldsOfStudy",
                "limit": RESULTS_PER_PAGE
            }
            if token:
                params["token"] = token
            
            try:
                response = self._make_request(BULK_SEARCH_ENDPOINT, params)
                batch = response.get("data", [])
                if not batch:
                    break
                    
                # Filter papers with abstracts
                batch_with_abstracts = [p for p in batch if p.get('abstract')]
                papers.extend(batch_with_abstracts)
                
                total_retrieved += len(batch)
                if progress_callback:
                    progress_callback(total_retrieved, len(batch_with_abstracts), len(papers))
                
                token = response.get("token")
                if not token:
                    break
                    
            except Exception as e:
                print(f"Error during search: {str(e)}")
                break
        
        return papers

    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        Get detailed metadata for a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
        
        Returns:
            Dictionary containing paper metadata
        """
        url = f"{PAPER_DETAILS_ENDPOINT}/{paper_id}"
        params = {"fields": ",".join(PAPER_FIELDS)}
        return self._make_request(url, params)

def process_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process and clean paper metadata, extracting generalised keywords from title+abstract using KeyBERT."""
    processed_papers = []
    keybert_model = KeyBERT()
    ngram_range = (1, 2)  # Prefer 1-2 word phrases for generality
    top_n = 10
    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        text = f"{title}. {abstract}".strip()
        keywords = []
        if text:
            candidates = keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=ngram_range,
                stop_words="english",
                top_n=top_n
            )
            # Filter: only keep keywords that are 1-3 words, not too specific
            seen = set()
            for kw, score in candidates:
                if 1 <= len(kw.split()) <= 3:
                    norm_kw = kw.lower().strip()
                    if norm_kw not in seen:
                        seen.add(norm_kw)
                        keywords.append(kw)
                if len(keywords) >= 7:
                    break
        processed_paper = {
            "title": paper.get("title", ""),
            "authors": [author.get("name", "") for author in paper.get("authors", [])],
            "year": paper.get("year", "N/A"),
            "abstract": paper.get("abstract", ""),
            "url": paper.get("url", ""),
            "keywords": keywords
        }
        processed_papers.append(processed_paper)
    return processed_papers

def main():
    """Main function to run the paper search."""
    try:
        # Get user input
        keywords_str = input("Enter keywords separated by spaces: ")
        keywords = keywords_str.strip().split()
        
        if not keywords:
            print("Error: No keywords provided")
            return
        
        # Initialize API client
        api = SemanticScholarAPI()
        
        # Search for papers
        print("\nSearching for papers...")
        papers = api.search_papers(keywords)
        
        if not papers:
            print("No papers found for the given keywords")
            return
        
        # Process papers
        print("\nProcessing paper metadata...")
        processed_papers = process_papers(papers)
        
        # Display summary
        print("\nSearch Summary:")
        print(f"Total papers found: {len(processed_papers)}")
        print(f"Keywords used: {', '.join(keywords)}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 