from find_papers import find_papers_by_keywords

def main():
    keywords = input("Enter keywords separated by spaces: ").strip().split()
    papers = find_papers_by_keywords(keywords, num_results=100)
    for i, paper in enumerate(papers, 1):
        print(f"\nPaper {i}:")
        print(f"URL: {paper['url']}")
        print(f"Title: {paper['title']}")
        print(f"Keywords: {paper['keywords']}")
        print(f"Abstract: {paper['abstract']}")

if __name__ == "__main__":
    main()