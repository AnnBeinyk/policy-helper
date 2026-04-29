# Databricks notebook source

# 01_bronze.py

from pyspark.sql.functions import (
    col,
    concat_ws,
    current_date,
    current_timestamp,
    lit,
    sha2,
)

# =========================
# CONFIG
# =========================

dbutils.widgets.text("storage_account", "sadlsdev")
dbutils.widgets.text("container", "beinykcontainer")
dbutils.widgets.text("catalog", "dbr_dev")
dbutils.widgets.text("schema", "beinyk_bronze")
dbutils.widgets.text("table_name", "policies_bronze")
dbutils.widgets.text("environment", "dev")

storage_account = dbutils.widgets.get("storage_account")
container = dbutils.widgets.get("container")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table_name = dbutils.widgets.get("table_name")
environment = dbutils.widgets.get("environment")

raw_path = f"abfss://{container}@{storage_account}.dfs.core.windows.net/raw/Policies/"
target_table = f"{catalog}.{schema}.{table_name}"

checkpoint_path = (
    f"abfss://{container}@{storage_account}.dfs.core.windows.net/"
    f"checkpoints/{environment}/bronze/{table_name}/"
)
schema_location = (
    f"abfss://{container}@{storage_account}.dfs.core.windows.net/"
    f"checkpoints/{environment}/bronze/{table_name}_schema/"
)

# =========================
# CREATE SCHEMA
# =========================

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# =========================
# READ WITH AUTO LOADER
# =========================

incoming_df = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", schema_location)
    .option("cloudFiles.inferColumnTypes", "true")
    .option("multiLine", "true")
    .load(raw_path)
)

# =========================
# SCHEMA VALIDATION
# =========================

required_columns = ["content", "source_file", "path"]
missing_columns = [c for c in required_columns if c not in incoming_df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns in Bronze input: {missing_columns}")

# =========================
# ADD METADATA AND STABLE KEYS
# =========================

bronze_df = (
    incoming_df.withColumn("source_path", col("_metadata.file_path"))
    .withColumn("ingested_at", current_timestamp())
    .withColumn("ingestion_date", current_date())
    .withColumn("environment", lit(environment))
    .withColumn("record_id", sha2(concat_ws("||", col("source_file"), col("path")), 256))
    .withColumn("content_hash", sha2(concat_ws("||", col("content"), col("category")), 256))
)

# =========================
# WRITE STREAM TO DELTA
# =========================

query = (
    bronze_df.writeStream.format("delta")
    .option("checkpointLocation", checkpoint_path)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(target_table)
)

query.awaitTermination()

# =========================
# DELTA TABLE PROPERTIES
# =========================

spark.sql(
    f"""
    ALTER TABLE {target_table}
    SET TBLPROPERTIES (
      delta.enableChangeDataFeed = true,
      delta.autoOptimize.optimizeWrite = true,
      delta.autoOptimize.autoCompact = true
    )
    """
)

# =========================
# JOB SUMMARY
# =========================

record_count = spark.table(target_table).count()
print(f"Bronze ingestion completed. Target table: {target_table}")
print(f"Total records in Bronze: {record_count}")
