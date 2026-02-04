# Claim Denial Explainability Copilot (RAG on Databricks)

## 1. What is this Project?

This project is a **Claim Denial Explainability Copilot** for a health insurance payer, built on **Databricks Lakehouse + Mosaic AI**.

It shows how a payer can use **Retrieval-Augmented Generation (RAG)** to answer questions like:

> “Why was this PCI claim denied with code HCD-014 under ACME_PPO_2025, and what documentation is needed to appeal?”

using **internal PDFs and policies** rather than generic internet knowledge.

The key idea:

- A **raw LLM alone** doesn’t know a payer’s internal denial codes, plan riders, and provider contracts.
- Those rules live in **long, private PDFs** that don’t fit into a single prompt and cannot be exposed publicly.
- This POC uses **RAG + Vector Search** so the model can **read the right pages from those PDFs on demand**, then explain the denial in clear language and cite the source.

The result is a **chat-style assistant** for business users (call-center staff, provider relations, nurses, etc.) that can explain claim decisions in seconds, grounded in the payer’s own documents.

---

## 2. Business problem & story

Today, when a claim is denied:

- A provider or member calls asking **“why?”**.
- Agents manually dig through:
  - Denial code manuals
  - Benefit handbooks / riders
  - Medical policy PDFs
  - Provider contracts
- This is slow, inconsistent, and error-prone.

A general LLM like ChatGPT can talk about prior authorization and medical necessity in general, but it **cannot**:

- Tell you **exactly what HCD-014 means** for *this* plan.
- Quote the **ACME_PPO_2025 benefit rider** language.
- Explain how **Medical Policy MP-PCI-001** treats elective vs emergent PCI.
- Show how the **provider contract ABC_CARD_1001** affects coverage.

This POC demonstrates how to:

1. **Ingest those internal PDFs into the Lakehouse.**  
2. **Index them with Vector Search.**  
3. **Build a RAG assistant** that:
   - Looks up the right snippets from those docs
   - Explains a denial in simple terms
   - Links back to the actual document sections used.

---

## 3. Data & domain (synthetic)

All data in this repo is **synthetic and non-PHI**.

The POC uses four example PDFs that mimic real payer content:

- `Denial_Code_Manual_NHP_2025.pdf`  
  - Internal denial codes such as **HCD-001, HCD-014, HCD-023, HCD-099**  
  - Each code has description, usage notes, and appeal guidance.
- `ACME_PPO_2025_Benefit_Rider.pdf`  
  - Plan-specific benefit and exclusion rules for **ACME_PPO_2025** (e.g., infertility, cardiac procedures, out-of-network).
- `Medical_Policy_MP_PCI_001.pdf`  
  - Clinical policy for **PCI with stent** (elective vs emergent criteria).
- `Provider_Contract_ABC_Cardiology_2024.pdf`  
  - Contract terms for provider **ABC_CARD_1001** (network status, reimbursement rules, carve-outs).

These PDFs are intentionally **longer than a typical LLM context window**, to illustrate why chunking + vector search are required.

---

## 4. High-level architecture

At a high level, the flow looks like this:

1. **Data sources**
   - Unstructured PDFs in a Unity Catalog **Volume**:
     - Denial code manual
     - Benefit rider
     - Medical policy
     - Provider contract

2. **Lakehouse data pipeline (Bronze → Silver → Gold)**  
   Built with Databricks notebooks:

   - **Bronze – Ingestion**
     - PDFs stored in `hc_rag.benefits_assistant.raw_docs` volume.
     - `pypdf` is used to extract text **per page**.
     - Output table: `hc_rag.benefits_assistant.bronze_pdf_pages`
       - `doc_id`, `file_path`, `page_num`, `total_pages`, `page_text`, `ingestion_ts`.

   - **Silver – Chunking & Enrichment**
     - Combine all pages for a given `doc_id` into a single `full_text`.
     - Add simple document metadata:
       - `doc_type` ∈ {`denial_manual`, `benefit_rider`, `medical_policy`, `provider_contract`, …}.
     - Split `full_text` into **overlapping chunks** (e.g., 1800 chars with 300-char overlap) to respect LLM context limits.
     - Output table: `hc_rag.benefits_assistant.silver_chunks`
       - `chunk_id`, `doc_id`, `doc_type`, `file_path`, `chunk`.

   - **Gold – Embeddings Store**
     - Use **Databricks Foundation Model embeddings** (e.g., `databricks-bge-large-en`) to embed each `chunk`.
     - Embeddings are obtained via a Databricks **serving endpoint**.
     - Output table: `hc_rag.benefits_assistant.gold_chunks_with_embeddings`
       - `chunk_id`, `doc_id`, `doc_type`, `file_path`, `chunk`, `embedding`.

3. **Vector Search**
   - A **Vector Search endpoint** hosts one or more indices.
   - Two styles are used:
     - A **self-managed embedding index** on `gold_chunks_with_embeddings` for experimentation in notebooks.
     - A **Databricks-managed embedding index** on `silver_chunks` (embeddings computed and updated by the platform) for use as a **Playground / Agent tool**.
   - The index stores vectors + metadata, and supports semantic similarity search over the chunks.

4. **RAG logic (notebooks)**
   - `retrieve_chunks(query)`:
     - Embed the user’s question using the same embedding model.
     - Call `VectorSearchClient.similarity_search()`.
     - Return the top-k chunks with `doc_id`, `doc_type`, `chunk`, and scores.
   - `rag_answer(question)`:
     - Uses `retrieve_chunks()` to build a contextual “evidence” block.
     - Calls a chat model (e.g., `databricks-meta-llama-3-3-70b-instruct`) via `ai_query`.
     - Prompt instructs the model to **only** answer using the retrieved context and to cite which docs were used.
     - Returns:
       - A natural-language explanation of the denial.
       - The set of chunks used to generate that answer.

5. **Chat assistant (Databricks AI Playground / Agent)**
   - In AI Playground, a **tools-enabled model** is configured with:
     - A **system prompt** describing the “Claim Denial Explainability Copilot” persona.
     - A **Vector Search tool** hooked to the managed index:
       - Tool description explains that it retrieves denial policies, benefit rules, and provider contract language.
   - When a business user asks a question in chat:
     - The agent automatically calls the Vector Search tool with an internal query like  
       “ACME_PPO_2025 elective PCI denial HCD-014”.
     - The tool returns the most relevant chunks from the indexed PDFs.
     - The model reads those chunks and generates an answer, referencing the actual plan names, denial codes, and policy IDs.

---

## 5. Repository structure

A typical layout for this POC:

```text
Claim-Denial-Explainability-Copilot/
  notebooks/
    01_setup_and_bronze_ingest.py        # create catalog/schema/volume + ingest PDFs → bronze_pdf_pages
    02_silver_chunking.py                # combine pages, chunk text, write silver_chunks
    03_gold_embeddings.py                # call Databricks FM embeddings → gold_chunks_with_embeddings
    04_vector_search_index.py            # create Vector Search endpoint + indices
    05_retrieve_chunks_demo.py           # retrieve_chunks() function, inspect top-k results
    06_rag_answer_demo.py                # rag_answer() that calls LLM with retrieved context
  src/
    (optional helpers, e.g. rag_utils/, not strictly required for the POC)
  README.md                              # this file

