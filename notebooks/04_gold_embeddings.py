# Databricks notebook source
# DBTITLE 1,Generate Gold Embeddings with Databricks Foundation Model
# Complete end-to-end pipeline for generating embeddings using Databricks Foundation Model API

from databricks.sdk import WorkspaceClient
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType, FloatType
from pyspark.sql import functions as F

# Configuration
catalog = "hc_rag"
schema = "benefits_assistant"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE {schema}")

# Initialize Databricks client for Foundation Model API
w = WorkspaceClient()

# 1) Read silver_chunks
silver_df = spark.table(f"{catalog}.{schema}.silver_chunks")

print("Silver row count:", silver_df.count())
display(silver_df.limit(5))

# 2) Collect to driver (OK for small POC)
rows = silver_df.collect()

# 3) For each chunk, call Databricks Foundation Model embeddings API
new_rows = []

for row in rows:
    text = row["chunk"]
    if text is None or text.strip() == "":
        embedding = []
    else:
        # Use Databricks Foundation Model API - databricks-bge-large-en (free, no API key needed)
        resp = w.serving_endpoints.query(
            name="databricks-bge-large-en",
            input=[text]  # Changed from 'inputs' to 'input'
        )
        embedding = resp.data[0].embedding

    new_rows.append((
        row["chunk_id"],
        row["doc_id"],
        row["doc_type"],
        row["file_path"],
        row["chunk"],
        embedding
    ))

# 4) Define schema for Gold table
schema_gold = StructType([
    StructField("chunk_id",   LongType(),   False),
    StructField("doc_id",     StringType(), False),
    StructField("doc_type",   StringType(), False),
    StructField("file_path",  StringType(), False),
    StructField("chunk",      StringType(), True),
    StructField("embedding",  ArrayType(FloatType()), True),
])

gold_df = spark.createDataFrame(new_rows, schema_gold)

# 5) Write Gold table
(
    gold_df.write
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.gold_chunks_with_embeddings")
)

print(f"✓ Successfully created table: {catalog}.{schema}.gold_chunks_with_embeddings")

# 6) Quick check
gold_check = spark.table(f"{catalog}.{schema}.gold_chunks_with_embeddings")
display(gold_check.limit(5))
print("Gold row count:", gold_check.count())