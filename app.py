import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Local imports
from pipeline import find_papers, filter_papers_by_keywords, rank_papers_by_content, get_pipeline_stats
from literature_summarizer import generate_summary

# Load environment variables
load_dotenv()

# --- 1. App State Initialization ---
def initialize_state():
    # Flags to track button clicks
    if 'start_search' not in st.session_state:
        st.session_state.start_search = False
    if 'generate_summary' not in st.session_state:
        st.session_state.generate_summary = False
    
    # Data storage
    if 'papers_topn' not in st.session_state:
        st.session_state.papers_topn = None
    if 'summary_text' not in st.session_state:
        st.session_state.summary_text = None
    
    # NEW: Add state variables for pipeline statistics
    if 'pipeline_stats' not in st.session_state:
        st.session_state.pipeline_stats = {}
    
    # User inputs that need to persist for callbacks
    if 'user_keywords' not in st.session_state:
        st.session_state.user_keywords = ""
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ""
    if 'top_n' not in st.session_state:
        st.session_state.top_n = 20
    if 'min_year' not in st.session_state:
        st.session_state.min_year = 2015
    if 'sort_by' not in st.session_state:
        st.session_state.sort_by = "Relevance Score"

initialize_state()

# --- 2. Callback Functions ---
def set_search_flag():
    st.session_state.start_search = True

def set_summary_flag():
    st.session_state.generate_summary = True

# --- Helper Functions ---
def get_author_names(authors):
    """Return a comma-separated string of author names."""
    return ', '.join(a['name'] if isinstance(a, dict) and 'name' in a else str(a) for a in authors)

def format_paper_card(paper, show_similarity=None):
    """Helper function to format paper information consistently."""
    title = paper.get('title', 'Untitled')
    year = paper.get('year', 'N/A')
    authors = get_author_names(paper.get('authors', []))
    keywords = paper.get('keywords', [])
    abstract = paper.get('abstract', '')
    url = paper.get('url', '#')
    
    keyword_tags = ' '.join([f"<span class='keyword-tag'>{k}</span>" for k in keywords]) if keywords else ''
    
    similarity_badge = ""
    if show_similarity:
        score = paper.get(show_similarity, 0)
        similarity_badge = f"<span class='similarity-badge'>Relevance: {score:.3f}</span>"
    
    card = f"""
    <div class='paper-card'>
        <h4>{title} ({year})</h4>
        {similarity_badge}
        <p><em>Authors:</em> {authors}</p>
        <div class='keywords-list'>{keyword_tags}</div>
        <p><small>{(abstract or 'No abstract available.')}</small></p>
        <p><a href='{url}' target='_blank'>🔗 View Paper</a></p>
    </div>
    """
    return card
    
# --- 3. UI Definition ---
# Define the entire UI first.
st.set_page_config(page_title="ScholarSift", page_icon="📚", layout="wide")
st.markdown("""
    <style>
    .app-title {
        font-size: 3rem;
        font-weight: 600;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .paper-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        background-color: #1e1e1e;
    }
    .similarity-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        background-color: #2e2e2e;
        margin-right: 0.5rem;
        color: #00ff00;
    }
    .keywords-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.5rem 0;
    }
    .keyword-tag {
        background-color: #2e2e2e;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        color: #00a0ff;
    }
    .intro-box {
        background-color: #262730;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        border: 1px solid #4a4d5d;
    }
    a {
        color: #00a0ff !important;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True) 

st.markdown("<h1 class='app-title'>📚 ScholarSift</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Intelligently discover and filter research papers</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("🔍 Search Settings")
    st.session_state.user_keywords = st.text_area("1. Enter Keywords:", placeholder="e.g., NLP, machine learning")
    
    st.subheader("Groq API Key")
    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        st.success("✅ Groq API Key loaded from .env file.")
        st.session_state.groq_api_key = env_key
    else:
        st.warning("Groq API Key not found.")
        st.session_state.groq_api_key = st.text_input("Please enter your key:", type="password")

    st.subheader("2. Configure Results")
    st.session_state.top_n = st.number_input("Top papers:", 1, 100, 20)
    st.session_state.min_year = st.number_input("Since year:", 1900, datetime.now().year, 2015)
    st.session_state.sort_by = st.selectbox("3. Sort Results By:", ["Relevance Score", "Year (Newest)", "Year (Oldest)"])
    
    st.button("🔍 Find Papers", on_click=set_search_flag, type="primary", use_container_width=True)

# --- 4. Controller Logic ---
if st.session_state.start_search:
    if st.session_state.user_keywords.strip():
        # Reset state for a new search
        st.session_state.papers_initial = None
        st.session_state.papers_keyword_filtered = None
        st.session_state.papers_topn = None
        st.session_state.summary_text = None
        st.session_state.pipeline_stats = {}

        with st.spinner("🚀 Step 1/4: Searching..."):
            papers = find_papers(st.session_state.user_keywords.split(','))
            st.session_state.pipeline_stats['initial_found'] = len(papers)
            st.session_state.papers_initial = papers # Store for inspection

        if papers:
            with st.spinner(f"📅 Step 2/4: Filtering papers since {st.session_state.min_year}..."):
                papers = [p for p in papers if p.get('year') and p.get('year') >= st.session_state.min_year]
                st.session_state.pipeline_stats['after_year_filter'] = len(papers)
                st.session_state.papers_keyword_filtered = papers

        if papers:
            with st.spinner("🎯 Step 3/4: Filtering by keyword relevance (top 30%)..."):
                papers = filter_papers_by_keywords(papers, st.session_state.user_keywords)
                st.session_state.pipeline_stats['after_keyword_filter'] = len(papers)
                st.session_state.papers_keyword_filtered = papers

        if papers:
            with st.spinner(f"🔥 Step 4/4: Applying advanced re-ranking for top {st.session_state.top_n}..."):
                papers = rank_papers_by_content(papers, st.session_state.user_keywords, st.session_state.top_n)
                
                # Final sort based on user choice
                if st.session_state.sort_by == "Year (Newest)":
                    papers.sort(key=lambda x: x.get('year', 0), reverse=True)
                elif st.session_state.sort_by == "Year (Oldest)":
                    papers.sort(key=lambda x: x.get('year', 0))
                
                st.session_state.papers_topn = papers
        else:
            st.session_state.papers_topn = [] # No results found
    else:
        st.warning("Please enter keywords to start a search.")
    st.session_state.start_search = False # Reset the flag

if st.session_state.generate_summary:
    if st.session_state.papers_topn:
        with st.spinner("✨ Generating AI summary with Groq..."):
            st.session_state.summary_text = generate_summary(st.session_state.papers_topn, st.session_state.groq_api_key)
    else:
        st.error("Cannot generate summary, no papers found.")
    st.session_state.generate_summary = False # Reset the flag

# --- 5. Display Logic ---
if st.session_state.papers_topn is not None:
    if not st.session_state.papers_topn:
        st.warning("Search complete. No papers found matching your criteria.")
    else:
        # --- Display Results ---
        papers_to_display = st.session_state.papers_topn
        stats = st.session_state.pipeline_stats
        st.success(f"📚 Process complete. Displaying top {len(papers_to_display)} papers.")

        # --- Display Pipeline Stats ---
        st.markdown("#### Search Funnel")
        col1, col2, col3 = st.columns(3)
        col1.metric("Papers Found", stats.get('initial_found', 0))
        col2.metric(f"Papers Since {st.session_state.min_year}", stats.get('after_year_filter', 0))
        col3.metric("Keyword-Filtered", stats.get('after_keyword_filter', 0))
        
        # --- NEW: Expanders for Intermediate Results ---
        with st.expander(f"🔍 View {st.session_state.pipeline_stats.get('after_year_filter', 0)} papers found since {st.session_state.min_year}"):
            for paper in st.session_state.get('papers_initial', []):
                # Note: Using a simplified card format for intermediate steps for clarity
                st.markdown(f"**{paper.get('title', 'No Title')}** ({paper.get('year', 'N/A')})")
                st.markdown(f"<small>Authors: {get_author_names(paper.get('authors', []))}</small>", unsafe_allow_html=True)
                st.markdown("---")

        with st.expander(f"🎯 View {st.session_state.pipeline_stats.get('after_keyword_filter', 0)} papers after keyword filtering"):
            for paper in st.session_state.get('papers_keyword_filtered', []):
                st.markdown(f"**{paper.get('title', 'No Title')}** ({paper.get('year', 'N/A')})")
                st.markdown(f"<small>Keyword Similarity: {paper.get('keyword_similarity', 0):.3f}</small>", unsafe_allow_html=True)
                st.markdown("---")

        st.markdown("---")
        st.header(f"🏆 Top {len(st.session_state.papers_topn)} Ranked Papers")
        
        main_tab1, main_tab2 = st.tabs(["📜 Research Papers", "✨ AI Summary"])

        # --- Tab 1: Paper Results and Export ---
        with main_tab1:
            for p in papers_to_display:
                st.markdown(format_paper_card(p, show_similarity='content_similarity'), unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("📥 Export Results")
            # Prepare data for export, ensuring native python types for JSON
            papers_for_export = []
            for p in st.session_state.papers_topn:
                paper_copy = p.copy()
                if 'content_similarity' in paper_copy:
                    paper_copy['content_similarity'] = float(paper_copy['content_similarity'])
                if 'keyword_similarity' in paper_copy:
                    paper_copy['keyword_similarity'] = float(paper_copy['keyword_similarity'])
                papers_for_export.append(paper_copy)

            df_export = pd.DataFrame(papers_for_export)
            st.download_button("Download CSV", df_export.to_csv(index=False), "results.csv")
            st.download_button("Download JSON", json.dumps(papers_for_export, indent=2), "results.json")

        # --- Tab 2: AI-Generated Summary ---
        with main_tab2:
            st.subheader("✨ AI-Generated Summary")
            st.button("Generate Summary", on_click=set_summary_flag)
            if st.session_state.summary_text:
                st.markdown(st.session_state.summary_text)
elif not st.session_state.start_search:
    st.info("👈 Enter your keywords in the sidebar and click 'Find Papers' to begin.") 