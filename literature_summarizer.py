from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

def _get_citation_key(paper: Dict, index: int) -> str:
    """Creates a human-readable citation key like [#1, Author et al., Year]."""
    authors = paper.get('authors', [])
    year = paper.get('year', 'N/A')
    
    # Use the index from the displayed list (1-based)
    paper_number = index + 1
    
    if not authors:
        first_author = "N.A."
    else:
        # Get the last name of the first author
        first_author = authors[0].get('name', '').split(' ')[-1]

    if len(authors) > 1:
        return f"[#{paper_number}, {first_author} et al., {year}]"
    
    return f"[#{paper_number}, {first_author}, {year}]"

def generate_summary(papers: List[Dict], api_key: str) -> str:
    """
    Generates an in-depth, academic-style literature review with detailed,
    numbered citations from a list of research papers.

    Args:
        papers: A list of paper dictionaries.
        api_key: The user's Groq API key.

    Returns:
        A string containing the AI-generated literature review.
    """
    if not api_key:
        return "Error: Groq API key is missing. Please enter it in the sidebar."
    if not papers:
        return "No papers were provided to summarize."

    # Use Groq's largest model for the highest quality synthesis
    model = ChatGroq(model="llama3-70b-8192", groq_api_key=api_key)

    # An even more advanced prompt for a detailed literature review
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a university professor and a world-class research analyst. Your task is to write an in-depth, "
            "analytical literature review based on the provided research papers. Your review must be well-structured, "
            "comprehensive, and several paragraphs long. \n\n"
            "Your structure should be as follows:\n"
            "1.  **Introduction:** Briefly introduce the overarching research domain and the key problems being addressed.\n"
            "2.  **Thematic Analysis:** Synthesize the papers' core themes, findings, and methodologies. Group similar papers "
            "and compare/contrast their approaches. You MUST dedicate at least 1-2 sentences to several of the most "
            "important papers to explain their specific contributions.\n"
            "3.  **Conclusion:** Summarize the collective insights and perhaps suggest potential future research directions.\n\n"
            "CRITICAL INSTRUCTION: When you reference any concept, finding, or method from a specific paper, "
            "you MUST use its full citation key, for example: `[#1, Author et al., Year]`, directly in the text."
        )),
        ("human", (
            "Here is the list of research papers. Please write your analytical literature review based on them:\n\n"
            "{papers_text}"
        ))
    ])
    
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    # Format the paper data with new, numbered citation keys
    papers_text = ""
    for i, paper in enumerate(papers):
        citation_key = _get_citation_key(paper, i)
        title = paper.get('title', 'No Title')
        abstract = paper.get('abstract', 'No Abstract Available')
        papers_text += f"Citation: {citation_key}\nTitle: {title}\nAbstract: {abstract}\n\n---\n\n"
    
    try:
        summary = chain.invoke({"papers_text": papers_text})
        return summary
    except Exception as e:
        return f"An error occurred while generating the summary: {e}" 