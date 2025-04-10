import os
import requests
from dotenv import load_dotenv

load_dotenv()

class PaperSearch:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("Missing SERPAPI_KEY. Please check your .env file.")

        self.base_url = "https://serpapi.com/search"

    def search_papers(self, query, num_results=5):
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }

        response = requests.get(self.base_url, params=params)

        if response.status_code != 200:
            raise Exception(f"SerpAPI request failed with status {response.status_code}")

        data = response.json()

        papers = []
        for result in data.get("organic_results", []):
            pub_info = result.get("publication_info", {})
            paper_info = {
                "title": result.get("title"),
                "link": result.get("link"),
                "snippet": result.get("snippet", "No abstract available."),
            }
            papers.append(paper_info)

        return papers

if __name__ == "__main__":
    searcher = PaperSearch()

    topic = input("Enter a research subtopic to search papers for: ")
    papers = searcher.search_papers(topic)

    print("\nTop Research Papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper['title']}")
        print(f"   Link: {paper['link']}")
        abstract = paper['snippet'][:250] + '...' if len(paper['snippet']) > 250 else paper['snippet']
        print(f"   Abstract: {abstract}")