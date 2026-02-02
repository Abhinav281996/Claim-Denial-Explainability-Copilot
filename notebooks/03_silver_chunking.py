# Databricks notebook source
# =========================
# STEP 3 – Build silver_chunks
# =========================

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

# 1) Set catalog & schema
catalog = "hc_rag"
schema = "benefits_assistant"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE {schema}")

# 2) Read Bronze table (pages)
bronze = spark.table(f"{catalog}.{schema}.bronze_pdf_pages")

# 3) Add doc_type based on doc_id pattern
doc_type_expr = (
    F.when(F.col("doc_id").like("%Denial_Code_Manual%"),   F.lit("denial_manual"))
     .when(F.col("doc_id").like("%Benefit_Rider%"),        F.lit("benefit_rider"))
     .when(F.col("doc_id").like("%Medical_Policy%"),       F.lit("medical_policy"))
     .when(F.col("doc_id").like("%Provider_Contract%"),    F.lit("provider_contract"))
     .otherwise(F.lit("other"))
)

bronze_with_type = bronze.withColumn("doc_type", doc_type_expr)

display(bronze_with_type.limit(5))

# 4) Combine all pages for each doc into one full_text
combined = (
    bronze_with_type
    .groupBy("doc_id", "file_path", "doc_type")
    .agg(
        F.concat_ws(
            "\n\n--- PAGE BREAK ---\n\n",
            F.collect_list(F.col("page_text"))
        ).alias("full_text")
    )
)

display(combined.limit(4))

# 5) Chunking function (simple char-based with overlap)
CHUNK_SIZE = 1800    # characters
CHUNK_OVERLAP = 300  # characters

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if text is None:
        return []
    text = text.strip()
    if text == "":
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap
        if start >= n:
            break
    return chunks

chunk_udf = F.udf(chunk_text, ArrayType(StringType()))

# 6) Apply chunking and explode into rows
chunked = (
    combined
    .withColumn("chunks", chunk_udf("full_text"))
    .withColumn("chunk", F.explode("chunks"))
    .withColumn("chunk_id", F.monotonically_increasing_id())
    .select(
        "chunk_id",
        "doc_id",
        "doc_type",
        "file_path",
        "chunk"
    )
)

display(chunked.limit(10))

# 7) Save as Silver table
(
    chunked.write
    .mode("overwrite")
    .saveAsTable(f"{catalog}.{schema}.silver_chunks")
)

# 8) Quick check
silver_check = spark.table(f"{catalog}.{schema}.silver_chunks")
display(silver_check.limit(5))
print("Silver row count:", silver_check.count())
