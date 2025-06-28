# Research Paper Finder & Ranker

## Technical Overview
This project implements a modular pipeline for **automated research paper discovery, filtering, and ranking** using state-of-the-art NLP and IR techniques. The system is designed for reproducibility, extensibility, and technical clarity.

---

## Pipeline Architecture

### 1. **Paper Retrieval (Step 1)**
- **API Used:** [Semantic Scholar API](https://api.semanticscholar.org/api-docs/graph)
- **Endpoint:** `/graph/v1/paper/search/bulk`
- **Query:** User-provided keywords are sent as the `query` parameter.
- **Fields:** The API returns metadata including `title`, `authors`, `year`, `abstract`, and (optionally) `fieldsOfStudy`.
- **Batching:** The pipeline retrieves up to 100 papers per query, using token-based pagination if needed.
- **Output:** A list of paper metadata dicts.

### 2. **Keyword Extraction (KeyBERT)**
- **Library:** [KeyBERT](https://github.com/MaartenGr/KeyBERT)
- **Input:** For each paper, the `title` and `abstract` are concatenated.
- **Model:** KeyBERT uses a BERT-based embedding model (by default, `all-MiniLM-L6-v2`) to generate document embeddings.
- **Extraction:** KeyBERT extracts the top N (configurable, e.g., 7) keywords/phrases from the concatenated text, using cosine similarity between candidate n-grams and the document embedding.
- **Output:** Each paper is augmented with a `keywords` field (list of strings).

### 3. **Semantic Filtering (Step 2)**
- **Embedding Model:** [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- **Process:**
    - The user query is embedded as a vector.
    - Each paper's extracted `keywords` are joined and embedded.
    - **Cosine similarity** is computed between the query embedding and each paper's keyword embedding.
    - **Ranking:** All papers are ranked by similarity; the top K (e.g., 50) are retained for the next stage.
- **Rationale:** This step ensures that only papers with high semantic relevance to the query are considered for final ranking.

### 4. **Content-Based Ranking (Step 3)**
- **Embedding Model:** [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- **Process:**
    - For each filtered paper, the `title`, `keywords`, and `abstract` are concatenated and embedded.
    - The user query is embedded as before.
    - **Cosine similarity** is computed between the query and each paper's full content embedding.
    - **Ranking:** The top N (e.g., 20) papers are selected and presented to the user, sorted by similarity score.

### 5. **(Optional) Topic Clustering**
- **Algorithm:** [BERTopic](https://github.com/MaartenGr/BERTopic) or similar topic modeling/clustering algorithm.
- **Input:** Embeddings of paper content (title+keywords+abstract).
- **Process:**
    - Papers are clustered in embedding space using HDBSCAN or UMAP+KMeans (as in BERTopic).
    - Each cluster is assigned a set of representative topic keywords.
- **Output:** Papers are annotated with cluster/topic labels, enabling topic-based exploration and filtering.

### 6. **User Interface**
- **Framework:** [Streamlit](https://streamlit.io/)
- **Features:**
    - User inputs query and configures number of top papers.
    - Results are displayed for each pipeline stage (100, 50, top N).
    - Paper metadata, similarity scores, and (optionally) cluster/topic labels are shown.
    - All computation is performed in-memory; no persistent storage is required.

---

## Data Flow Summary
1. **User Query** → Semantic Scholar `/paper/search/bulk` → **Raw Paper Metadata**
2. **Raw Paper Metadata** → KeyBERT → **Paper Keywords**
3. **User Query + Paper Keywords** → sentence-transformers + cosine similarity → **Top 50 Papers**
4. **User Query + Paper Content** → sentence-transformers + cosine similarity → **Top N Papers**
5. **(Optional) Paper Content Embeddings** → BERTopic → **Topic Clusters**

---

## Technologies & Methods
- **Semantic Scholar API**: `/graph/v1/paper/search/bulk` endpoint for scalable, metadata-rich paper retrieval.
- **KeyBERT**: BERT-based unsupervised keyword/keyphrase extraction from title+abstract.
- **sentence-transformers**: State-of-the-art sentence/document embedding for semantic similarity and ranking.
- **Cosine Similarity**: Vector-based similarity metric for ranking and filtering.
- **BERTopic**: Topic modeling and clustering for unsupervised topic discovery (optional).
- **Streamlit**: Interactive, real-time web UI for pipeline configuration and result exploration.

---

## Setup & Usage
1. **Install Requirements**
   ```sh
   pip install -r requirements.txt
   ```
2. **Run the App**
   ```sh
   streamlit run app.py
   ```
3. **Configure and Search**
   - Enter your query and select the number of top papers.
   - View results at each pipeline stage, including similarity scores and (optionally) topic clusters.

---

## Example Query
- Query: `NLP, LLM, ASR, TTS, sentiment analysis`
- Output: Top N most semantically relevant papers, with extracted keywords, similarity scores, and (optionally) topic clusters.

---

## FAQ
**Q: Which Semantic Scholar endpoint is used?**  
A: `/graph/v1/paper/search/bulk` for efficient, paginated paper search.

**Q: How are keywords extracted?**  
A: Using KeyBERT on the concatenated title and abstract of each paper.

**Q: How is relevance determined?**  
A: By computing cosine similarity between user query and paper embeddings (keywords and full content) using sentence-transformers.

**Q: Can I explore topics?**  
A: Yes, optional clustering (e.g., BERTopic) groups papers by topic for further exploration.

---

## Credits
- [Semantic Scholar API](https://www.semanticscholar.org/product/api)
- [KeyBERT](https://github.com/MaartenGr/KeyBERT)
- [Sentence-Transformers](https://www.sbert.net/)
- [BERTopic](https://github.com/MaartenGr/BERTopic)
- [Streamlit](https://streamlit.io/) 