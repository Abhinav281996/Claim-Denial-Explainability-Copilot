# Databricks notebook source
# MAGIC %sql
# MAGIC -- Use or create catalog for this POC
# MAGIC CREATE CATALOG IF NOT EXISTS hc_rag;
# MAGIC
# MAGIC USE CATALOG hc_rag;
# MAGIC
# MAGIC -- Schema for this use case
# MAGIC CREATE SCHEMA IF NOT EXISTS benefits_assistant;
# MAGIC
# MAGIC USE benefits_assistant;
# MAGIC
# MAGIC -- Volume to hold the internal PDFs
# MAGIC CREATE VOLUME IF NOT EXISTS raw_docs;
# MAGIC

# COMMAND ----------

