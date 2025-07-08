# 📚 LLM Research Bot

A powerful, interactive tool to search, filter, rank, and visualize research papers using advanced language models and the Semantic Scholar API.

---

## 🚀 Key Features

- **Semantic Search:** Find the most relevant research papers for your query using state-of-the-art language models.
- **Step-by-Step Filtering:** See how your results are refined at each stage (initial search, year filter, keyword filter, content ranking).
- **Beautiful UI:** Modern, user-friendly interface with clear progress panels, metrics, and expandable details.
- **Paper Tree Visualization:** Explore not just how papers relate to your query, but also how they relate to each other.
- **AI-Generated Summaries:** Get a literature review summary of your top papers.
- **Export Results:** Download your results as CSV or JSON for further analysis or sharing.
- **Transparent Pipeline:** Every step is visible, so you can see exactly how your results are refined.

---

## 🛠️ How It Works

1. **User enters keywords** (e.g., "Natural Language Processing, Sentiment Analysis").
2. **Papers are fetched** from Semantic Scholar using those keywords.
3. **Pipeline refines results** through several steps:
   - Initial search (broad retrieval)
   - Year filter (removes old papers)
   - Keyword similarity filter (semantic filtering)
   - Content-based ranking (deep semantic ranking)
4. **Results are displayed** with clear step panels, metrics, and expandable lists.
5. **Visualizations** show both the relevance to your query and inter-paper relationships.
6. **Summarization**: Generate an AI-powered literature review of your top results.

---

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kaushik-2005/LLM-Research-Bot.git
   cd llm-research-bot
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up your environment:**
   - Add your Semantic Scholar API key and Groq API key to a `.env` file:
     ```
     SEMANTIC_SCHOLAR_API_KEY=your_key_here
     GROQ_API_KEY=your_groq_key_here
     ```
4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## 🖥️ Usage

- Enter your keywords in the sidebar and configure the number of top papers.
- Click **Run Pipeline** to start the search and filtering process.
- Explore each step's results using the step panels and expanders.
- Switch between tabs to view:
  - **Research Papers:** Full details and cards for each top paper.
  - **AI Summary:** An LLM-generated literature review.
  - **Paper Tree:** Visualize relationships between your query and the top papers, as well as between the papers themselves.
- Download your results as CSV or JSON for further analysis.

---

## 🔎 Pipeline Steps Explained

### Step 1: Initial Search
- Fetches up to 1000 papers matching your keywords from Semantic Scholar.
- Shows how many papers were found.
- You can expand to see all initial results.

### Step 2: Year & Keyword Filtering
- Filters out papers before your chosen year (e.g., since 2015).
- Ranks papers by keyword similarity to your query using sentence-transformer embeddings.
- You can expand to see the filtered results.

### Step 3: Content-Based Ranking
- Uses a cross-encoder model to deeply compare your query to each paper's content (title + abstract).
- Selects the top N most relevant papers (e.g., top 20).
- These are shown as detailed cards with all metadata.

### Step 4: Visualization & Export
- View results as cards, download them, or explore relationships in the Paper Tree.
- Generate an AI-powered summary of your top papers.

---

## 📊 Metrics & Search Funnel

- At the top of the Research Papers tab, you'll see key metrics:
  - **Papers Found:** Total papers retrieved from Semantic Scholar.
  - **Papers Since [Year]:** Papers published since your chosen year.
  - **Keyword-Filtered:** Papers remaining after keyword similarity filtering.
- These metrics help you understand how your search is refined at each step.

---

## 🌳 Paper Tree & Inter-Paper Connections

- The Paper Tree shows your query as the root and each top paper as a child node.
- **Orange edges** between papers indicate high content similarity (using the same cross-encoder model as for ranking).
- This helps you see not just which papers are relevant, but which are closely related to each other.
- Node size is proportional to the paper's relevance score.
- Hover over a node to see the paper's title, year, and score.

### How Inter-Paper Edges Are Computed
- For each pair of top-k papers, the app computes the content similarity between their title+abstracts using the cross-encoder model.
- If the similarity is above a threshold (e.g., 0.8), an orange edge is drawn between those two papers.
- This visualizes clusters or groups of highly related papers within your results.

---

## 🏷️ Understanding Scores

### Relevance Score (on Paper Cards)
- The **Relevance** score shown on each paper card is the similarity between your search query and the paper, computed using a cross-encoder model.
- This measures how relevant the paper is to your search.

### Inter-Paper Similarity Score (Paper Tree Edges)
- In the Paper Tree visualization, orange edges between papers indicate high content similarity between two papers, also computed using the cross-encoder model.
- This measures how similar the content of two papers is to each other, not to your query.

### Why the Scores May Differ
- The **Relevance** score is always between the query and a paper.
- The **Inter-Paper Similarity** score is between two different papers.
- The cross-encoder model may output scores on different scales for these two types of comparisons.
- For example, relevance scores might be in the range 0–10, while inter-paper similarity might be between 0 and 1, or even negative values, depending on the model.

### How to Interpret
- **Relevance**: Higher means the paper is more relevant to your search.
- **Inter-Paper Edge**: An edge means the two papers are highly similar in content (above a set threshold).
- The threshold for drawing an edge is set based on the similarity score (e.g., >0.8).

### Note
- It is normal for these scores to be on different scales.
- If you want to adjust the threshold for inter-paper edges to match the scale of the relevance score, you can do so in the code.

---

## 🧠 AI-Generated Literature Review

- The app can generate a multi-paragraph, academic-style summary of your top papers using an LLM (via Groq API).
- The summary includes:
  - Introduction to the research domain
  - Thematic analysis of the papers
  - Conclusion and future directions
- Each reference in the summary is clearly cited with a unique key (e.g., [#1, Author et al., Year]).

---

## ⚙️ Customization & Advanced Usage

- **Adjust thresholds** for filtering and inter-paper edges in `app.py`.
- **Change models** or add new visualization features as needed.
- **API keys** and other settings can be managed in `.env`.
- **Add new pipeline steps** or modify existing ones in `pipeline.py`.
- **Extend the UI** with new tabs, metrics, or export options.

---

## 🧩 Example Use Case

Suppose you want to find the most relevant papers on "Natural Language Processing, Sentiment Analysis" since 2015:

1. Enter your keywords and set the year to 2015.
2. The app finds 3000+ papers, filters to 900+ after keyword similarity, and ranks the top 20 by content.
3. You can:
   - Expand to see all papers at each step
   - View detailed cards for the top 20
   - See which papers are most closely related in the Paper Tree
   - Download your results
   - Generate an AI-powered summary

---

## ❓ Troubleshooting & FAQ

- **Why do some papers have no abstract?**  
  Not all papers in Semantic Scholar have abstracts available.
- **Why are some scores different?**  
  See the "Understanding Scores" section above.
- **How can I add new features?**  
  The code is modular—see `pipeline.py` and `app.py` for extension points.
- **How do I change the similarity threshold for inter-paper edges?**  
  Edit the `SIM_THRESHOLD` value in the Paper Tree tab code.
- **Can I use a different LLM or embedding model?**  
  Yes! Swap out the model in `pipeline.py` or `literature_summarizer.py`.

---

## 🙏 Credits & References

- [Semantic Scholar API](https://www.semanticscholar.org/product/api)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Pyvis](https://pyvis.readthedocs.io/) / [streamlit-agraph](https://github.com/ChrisDelClea/streamlit-agraph)
- [KeyBERT](https://github.com/MaartenGr/KeyBERT)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/)

---

## 📬 Contact & Contributions

- Found a bug? Have a feature request? Open an issue or pull request!
- For questions, contact the maintainer at [kaushik.kesanapalli@gmail.com].

---

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. 
