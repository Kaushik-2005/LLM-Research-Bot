import json
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from paper_search import SemanticScholarAPI
import math

# Step 1: Find 100 papers
def find_papers(keywords: List[str], progress_callback=None) -> List[Dict[str, Any]]:
    """
    Use SemanticScholarAPI to find all available papers with abstracts for the given keywords.
    Returns a list of paper dicts with title, abstract, keywords, etc.
    
    Args:
        keywords: List of search keywords
        progress_callback: Optional callback function to update progress
    """
    api = SemanticScholarAPI()
    papers = api.search_papers(keywords, progress_callback=progress_callback)
    return papers

# Step 2: Filter to top_k most relevant papers by keyword similarity (ranking only)
def filter_papers_by_keywords(papers: List[Dict[str, Any]], user_query: str, percentage: float = 0.3) -> List[Dict[str, Any]]:
    """
    Rank all papers by semantic similarity between user query and paper keywords.
    Returns the top percentage of most similar papers.
    
    Args:
        papers: List of papers to filter
        user_query: User's search query
        percentage: Percentage of papers to keep (default: 0.3 = 30%)
    """
    if not papers:
        return []
        
    model = SentenceTransformer('all-MiniLM-L6-v2')
    user_emb = model.encode(user_query, convert_to_tensor=True)
    scored = []
    
    for paper in papers:
        keywords = paper.get('keywords', [])
        kw_text = ' '.join(keywords)
        paper_emb = model.encode(kw_text, convert_to_tensor=True) if kw_text else None
        sim = util.cos_sim(user_emb, paper_emb).item() if paper_emb is not None else float('-inf')
        paper['keyword_similarity'] = sim
        scored.append(paper)
    
    # Sort by similarity and keep top percentage
    scored.sort(key=lambda x: x['keyword_similarity'], reverse=True)
    keep_count = math.ceil(len(papers) * percentage)
    return scored[:keep_count]

# Step 3: Rank and select top_n by full content similarity
def rank_papers_by_content(papers: List[Dict[str, Any]], user_query: str, top_n: int = 20) -> List[Dict[str, Any]]:
    """
    Re-ranks papers using a more powerful Cross-Encoder model for higher accuracy.
    This is much more accurate than cosine similarity as it performs full-attention over the query and text.
    
    Args:
        papers: A list of pre-filtered papers.
        user_query: The user's search query.
        top_n: The final number of papers to return.
        
    Returns:
        The top_n most relevant papers, sorted by the Cross-Encoder score.
    """
    if not papers:
        return []
        
    # Using a Cross-Encoder model specifically trained for re-ranking tasks.
    # It will be downloaded on the first run.
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # The Cross-Encoder needs pairs of [query, document]
    sentence_pairs = []
    for paper in papers:
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        # We use the title and abstract for the most relevant text.
        text_to_rank = f"{title}. {abstract}".strip()
        sentence_pairs.append([user_query, text_to_rank])
        
    # Predict the scores for all pairs. show_progress_bar is for console, but useful.
    scores = cross_encoder.predict(sentence_pairs, show_progress_bar=True)
    
    # Add scores to each paper object
    for i in range(len(papers)):
        papers[i]['content_similarity'] = scores[i]

    # Sort by the new cross-encoder score in descending order
    papers.sort(key=lambda x: x.get('content_similarity', float('-inf')), reverse=True)
    
    return papers[:top_n]

def get_pipeline_stats(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about the paper pipeline.
    
    Args:
        papers: List of papers
        
    Returns:
        Dictionary containing statistics about the papers
    """
    stats = {
        "total_papers": len(papers),
        "with_abstract": len([p for p in papers if p.get('abstract')]),
        "with_keywords": len([p for p in papers if p.get('keywords')]),
        "year_range": {
            "min": min((p.get('year', 9999) for p in papers if p.get('year')), default=None),
            "max": max((p.get('year', 0) for p in papers if p.get('year')), default=None)
        }
    }
    return stats 