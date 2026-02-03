# Databricks notebook source
# DBTITLE 1,Cell 1
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()
# MAGIC
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC import pandas as pd
# MAGIC
# MAGIC # -------------------------------------------------------------------
# MAGIC # 1) Config – align with what we've already used
# MAGIC # -------------------------------------------------------------------
# MAGIC catalog = "hc_rag"
# MAGIC schema = "benefits_assistant"
# MAGIC index_name = f"{catalog}.{schema}.claim_denial_rag_index"
# MAGIC
# MAGIC # (Make sure notebook is using the right catalog & schema)
# MAGIC spark.sql(f"USE CATALOG {catalog}")
# MAGIC spark.sql(f"USE {schema}")
# MAGIC
# MAGIC # -------------------------------------------------------------------
# MAGIC # 2) Init clients
# MAGIC # -------------------------------------------------------------------
# MAGIC w = WorkspaceClient()
# MAGIC vs_client = VectorSearchClient()
# MAGIC
# MAGIC index = vs_client.get_index(index_name=index_name)
# MAGIC
# MAGIC print("✅ Loaded Vector Search index:", index_name)
# MAGIC
# MAGIC # -------------------------------------------------------------------
# MAGIC # 3) Helper: get query embedding using the SAME model as step 4
# MAGIC # -------------------------------------------------------------------
# MAGIC def get_query_embedding(text: str):
# MAGIC     """
# MAGIC     Turn a question into an embedding using databricks-bge-large-en
# MAGIC     (same model used for the gold table).
# MAGIC     """
# MAGIC     if text is None or text.strip() == "":
# MAGIC         return []
# MAGIC
# MAGIC     resp = w.serving_endpoints.query(
# MAGIC         name="databricks-bge-large-en",
# MAGIC         input=[text]   # list of strings for embeddings endpoint
# MAGIC     )
# MAGIC
# MAGIC     # For BGE embedding endpoint, the embedding is in resp.data[0].embedding
# MAGIC     embedding = resp.data[0].embedding
# MAGIC     return embedding
# MAGIC
# MAGIC # -------------------------------------------------------------------
# MAGIC # 4) Helper: parse similarity_search() raw results into a DataFrame
# MAGIC # -------------------------------------------------------------------
# MAGIC def parse_search_results(raw_results):
# MAGIC     """
# MAGIC     Convert Databricks Vector Search raw response into a list of dicts,
# MAGIC     then into a pandas DataFrame for easy display.
# MAGIC     """
# MAGIC     try:
# MAGIC         data_array = raw_results["result"]["data_array"]
# MAGIC         columns = [c["name"] for c in raw_results["manifest"]["columns"]]
# MAGIC         rows = [dict(zip(columns, row)) for row in data_array]
# MAGIC         return pd.DataFrame(rows)
# MAGIC     except KeyError as e:
# MAGIC         print("Unexpected result format from similarity_search:", e)
# MAGIC         print("Raw results:", raw_results)
# MAGIC         return pd.DataFrame()
# MAGIC
# MAGIC # -------------------------------------------------------------------
# MAGIC # 5) Core: semantic retrieval function
# MAGIC # -------------------------------------------------------------------
# MAGIC def retrieve_chunks(query: str, k: int = 5):
# MAGIC     """
# MAGIC     1) Embed query text
# MAGIC     2) Call vector_search index.similarity_search with query_vector
# MAGIC     3) Return a pandas DataFrame of top-k chunks
# MAGIC     """
# MAGIC     print(f"\n🔎 Query: {query}\n")
# MAGIC
# MAGIC     query_emb = get_query_embedding(query)
# MAGIC     print("Embedding dimension:", len(query_emb))
# MAGIC
# MAGIC     results = index.similarity_search(
# MAGIC         query_vector=query_emb,
# MAGIC         columns=["chunk_id", "doc_id", "doc_type", "file_path", "chunk"],
# MAGIC         num_results=k,
# MAGIC         debug_level=1  # prints latency info in results["debug_info"]
# MAGIC     )
# MAGIC
# MAGIC     df = parse_search_results(results)
# MAGIC
# MAGIC     # Reorder columns (score is typically appended at the end)
# MAGIC     cols = df.columns.tolist()
# MAGIC     # Try to put score first if present
# MAGIC     if "score" in cols:
# MAGIC         cols = ["score"] + [c for c in cols if c != "score"]
# MAGIC
# MAGIC     df = df[cols]
# MAGIC     return df
# MAGIC
# MAGIC # -------------------------------------------------------------------
# MAGIC # 6) Try a test query
# MAGIC # -------------------------------------------------------------------
# MAGIC test_query = """
# MAGIC Why would a PCI (stent) claim be denied for ACME PPO 2025
# MAGIC if it was done electively without prior authorization?
# MAGIC """
# MAGIC
# MAGIC df_results = retrieve_chunks(test_query, k=5)
# MAGIC display(df_results)