# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import functions as F

catalog = "hc_rag"
schema = "benefits_assistant"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE {schema}")


# COMMAND ----------

vs_client = VectorSearchClient()

vs_endpoint_name = "hc_rag_claims_vs_endpoint"

# Create endpoint (if it already exists, you can ignore the error)
try:
    vs_client.get_endpoint(name=vs_endpoint_name)
    print(f"Vector Search endpoint already exists: {vs_endpoint_name}")
except Exception:
    vs_client.create_endpoint(
        name=vs_endpoint_name,
        endpoint_type="STANDARD"  # good for this POC
    )
    print(f"Created Vector Search endpoint: {vs_endpoint_name}")


# COMMAND ----------

gold_table = f"{catalog}.{schema}.gold_chunks_with_embeddings"

gold_df = spark.table(gold_table)
print("Gold rows:", gold_df.count())

first_emb = gold_df.select("embedding").limit(1).collect()[0][0]
embedding_dim = len(first_emb)
print("Embedding dimension:", embedding_dim)


# COMMAND ----------

# DBTITLE 1,Enable Change Data Feed on Gold Table
# Enable Change Data Feed (required for Vector Search delta sync)
spark.sql(f"""
  ALTER TABLE {gold_table}
  SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

print(f"✓ Change Data Feed enabled on {gold_table}")

# COMMAND ----------

# DBTITLE 1,Cell 6
index_name = f"{catalog}.{schema}.claim_denial_rag_index"

# Try to create index, or get it if it already exists
try:
    index = vs_client.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        source_table_name=gold_table,
        index_name=index_name,
        pipeline_type="TRIGGERED",          # manual sync for now
        primary_key="chunk_id",
        embedding_dimension=embedding_dim,
        embedding_vector_column="embedding",
        # Optional: limit synced columns if you want
        columns_to_sync=["chunk_id", "doc_id", "doc_type", "file_path", "chunk", "embedding"]
    )
    print("Created index:", index.name)
except Exception as e:
    if "already exists" in str(e):
        index = vs_client.get_index(endpoint_name=vs_endpoint_name, index_name=index_name)
        print(f"Index already exists: {index.name}")
    else:
        raise

# COMMAND ----------

# DBTITLE 1,Cell 7
import time

# Wait for endpoint to be ready
print("Waiting for vector search endpoint to be ready...")
while True:
    endpoint = vs_client.get_endpoint(vs_endpoint_name)
    status = endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"Endpoint status: {status}")
    
    if status == "ONLINE":
        print("Endpoint is ready!")
        break
    elif status in ["OFFLINE", "PROVISIONING"]:
        print("Waiting 30 seconds...")
        time.sleep(30)
    else:
        print(f"Unexpected status: {status}. Waiting 30 seconds...")
        time.sleep(30)

# After creating the index, trigger the first sync
index.sync()
print("Triggered initial sync")

# COMMAND ----------

idx = vs_client.get_index(index_name=index_name)
print("Index status:", idx.describe())
