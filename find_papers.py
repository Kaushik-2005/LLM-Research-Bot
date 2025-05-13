import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv(dotenv_path="C:/Users/kaush/B.Tech/PROJECTS/LLM-Research-Bot/.env")
print("Current working directory:", os.getcwd())
print("EXA_API_KEY loaded:", os.getenv("EXA_API_KEY"))

try:
    from exa_py import Exa
except ImportError:
    raise ImportError("exa_py not installed. Please install using pip install exa_py")

def find_papers_by_keywords(keywords: List[str], num_results: int = 100) -> List[Dict[str, str]]:
    """
    Given a list of keywords, search arXiv for relevant papers and return a list of dicts with url, title, keywords, and abstract.
    """
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY not set. Please set the EXA_API_KEY environment variable.")
    exa = Exa(api_key)
    query = ', '.join(keywords)
    search_results = exa.search_and_contents(
        query,
        num_results=num_results,
        include_domains=["arxiv.org"]
    )
    urls = [result.url for result in search_results.results]
    titles = [result.title for result in search_results.results]
    # Get keywords and abstract for each paper
    keywords_results = exa.get_contents(
        urls,
        summary={
            "enabled": True,
            "query": "List the main keywords of the paper"
        }
    )
    abstract_results = exa.get_contents(
        urls,
        summary={
            "enabled": True,
            "query": "Give me the abstract of the paper"
        }
    )
    papers = []
    for i, url in enumerate(urls):
        paper = {
            "url": url,
            "title": titles[i],
            "keywords": keywords_results.results[i].summary,
            "abstract": abstract_results.results[i].summary
        }
        papers.append(paper)
    return papers

if __name__ == "__main__":
    keywords_str = input("Enter keywords separated by spaces: ")
    keywords = keywords_str.strip().split()
    try:
        papers = find_papers_by_keywords(keywords)
        for i, paper in enumerate(papers, 1):
            # print(f"\nPaper {i}:")
            # print(f"URL: {paper['url']}")
            print(f"Title:{i} {paper['title']}")
            # print(f"Keywords: {paper['keywords']}")
            # print(f"Abstract: {paper['abstract']}")
    except Exception as e:
        print(f"Error: {e}")