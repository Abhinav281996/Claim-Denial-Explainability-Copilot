# Databricks notebook source
# DBTITLE 1,Cell 1
# 👉 Set these to what you actually used in Step 1
catalog = "hc_rag"   # Using the existing catalog
schema = "benefits_assistant"

volume_path = f"/Volumes/{catalog}/{schema}/raw_docs"

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE {schema}")

print("Volume path:", volume_path)

# COMMAND ----------

# DBTITLE 1,Cell 2
# Only install if not already available
try:
    import pypdf
    print("pypdf already installed")
except ImportError:
    %pip install pypdf
    dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 3
from pypdf import PdfReader
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import functions as F
import os

# Re-define variables after Python restart
catalog = "hc_rag"
schema = "benefits_assistant"
volume_path = f"/Volumes/{catalog}/{schema}/raw_docs"

# Make sure we're in the right catalog/schema again after restart
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE {schema}")

rows = []

# Walk the Unity Catalog volume and read every PDF
for root, dirs, files in os.walk(volume_path):
    for f in files:
        if not f.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(root, f)
        doc_id = os.path.splitext(f)[0]  # e.g. ACME_PPO_2025_Benefit_Rider

        reader = PdfReader(full_path)
        num_pages = len(reader.pages)

        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text() or ""

            rows.append(
                (
                    doc_id,
                    full_path,
                    page_num,
                    num_pages,
                    text,
                )
            )

# Define schema for the Bronze table
schema_bronze = StructType([
    StructField("doc_id", StringType(), False),
    StructField("file_path", StringType(), False),
    StructField("page_num", IntegerType(), False),
    StructField("total_pages", IntegerType(), False),
    StructField("page_text", StringType(), True),
])

bronze_df = (
    spark.createDataFrame(rows, schema_bronze)
         .withColumn("ingestion_ts", F.current_timestamp())
)

# Write as a managed Delta table in your schema
(
    bronze_df.write
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable("bronze_pdf_pages")    # fully qualified: f"{catalog}.{schema}.bronze_pdf_pages"
)

display(spark.table("bronze_pdf_pages").limit(10))