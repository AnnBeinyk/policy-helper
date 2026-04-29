# Databricks notebook / 02_silver.py

from pyspark.sql import Window
from pyspark.sql.functions import (
    col,
    concat_ws,
    current_timestamp,
    length,
    lit,
    posexplode,
    row_number,
    sha2,
    udf,
)
from pyspark.sql.types import ArrayType, StringType

# =========================
# CONFIG
# =========================

dbutils.widgets.text("catalog", "dbr_dev")
dbutils.widgets.text("bronze_schema", "beinyk_bronze")
dbutils.widgets.text("silver_schema", "beinyk_silver")
dbutils.widgets.text("bronze_table_name", "policies_bronze")
dbutils.widgets.text("silver_table_name", "policy_chunks")

catalog = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")
silver_schema = dbutils.widgets.get("silver_schema")
bronze_table_name = dbutils.widgets.get("bronze_table_name")
silver_table_name = dbutils.widgets.get("silver_table_name")

bronze_table = f"{catalog}.{bronze_schema}.{bronze_table_name}"
silver_table = f"{catalog}.{silver_schema}.{silver_table_name}"

# =========================
# CREATE SCHEMA
# =========================

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{silver_schema}")

# =========================
# CHUNK FUNCTION
# =========================

def split_text(text, chunk_size=500, overlap=100):
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step

    return chunks


split_udf = udf(lambda text: split_text(text), ArrayType(StringType()))

# =========================
# LOAD LATEST BRONZE VERSION
# =========================

bronze_df = spark.table(bronze_table)

required_columns = [
    "record_id",
    "content_hash",
    "content",
    "category",
    "source_file",
    "path",
    "ingested_at",
]
missing_columns = [c for c in required_columns if c not in bronze_df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns in Bronze table: {missing_columns}")

latest_window = Window.partitionBy("record_id").orderBy(col("ingested_at").desc())

latest_bronze_df = (
    bronze_df.withColumn("bronze_rank", row_number().over(latest_window))
    .filter(col("bronze_rank") == 1)
    .drop("bronze_rank")
)

# =========================
# BUILD CHUNKS
# =========================

chunked_df = (
    latest_bronze_df.withColumn("chunks", split_udf(col("content")))
    .select(
        "record_id",
        "content_hash",
        "category",
        "source_file",
        "path",
        "ingested_at",
        posexplode(col("chunks")).alias("chunk_index", "chunk"),
    )
    .filter(length(col("chunk")) > 50)
)

new_chunks_df = (
    chunked_df.withColumn("chunk_length", length(col("chunk")))
    .withColumn(
        "chunk_key",
        sha2(
            concat_ws(
                "||",
                col("record_id"),
                col("content_hash"),
                col("chunk_index").cast("string"),
            ),
            256,
        ),
    )
    .select(
        "chunk_key",
        "record_id",
        "chunk_index",
        "chunk",
        "chunk_length",
        "category",
        "source_file",
        "path",
        "content_hash",
        "ingested_at",
    )
)

# =========================
# DATA QUALITY CHECKS
# =========================

duplicate_chunk_count = (
    new_chunks_df.groupBy("chunk_key").count().filter(col("count") > 1).count()
)

if duplicate_chunk_count > 0:
    raise ValueError(f"Duplicate chunk_key detected: {duplicate_chunk_count}")

# =========================
# CHECK IF SILVER EXISTS
# =========================

table_exists = spark.catalog.tableExists(silver_table)

# =========================
# FIRST RUN - FULL BUILD
# =========================

if not table_exists:
    print("First run: building full Silver table with SCD Type 2 fields")

    final_df = (
        new_chunks_df.withColumn("processed_at", current_timestamp())
        .withColumn("valid_from", current_timestamp())
        .withColumn("valid_to", lit(None).cast("timestamp"))
        .withColumn("is_current", lit(True))
    )

    (
        final_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(silver_table)
    )

    spark.sql(
        f"""
        ALTER TABLE {silver_table}
        SET TBLPROPERTIES (
          delta.enableChangeDataFeed = true,
          delta.autoOptimize.optimizeWrite = true,
          delta.autoOptimize.autoCompact = true
        )
        """
    )

    print("Silver table created")

# =========================
# INCREMENTAL RUN - SCD TYPE 2
# =========================

else:
    print("Incremental run: applying SCD Type 2 changes")

    current_silver_df = (
        spark.table(silver_table)
        .filter(col("is_current") == True)
        .select("record_id", "content_hash")
        .distinct()
    )

    changed_documents_df = (
        latest_bronze_df.alias("new")
        .join(current_silver_df.alias("old"), on="record_id", how="left")
        .filter(
            col("old.content_hash").isNull()
            | (col("new.content_hash") != col("old.content_hash"))
        )
        .select("new.record_id", "new.content_hash")
        .distinct()
    )

    changed_count = changed_documents_df.count()

    if changed_count == 0:
        print("No changes detected. Silver table is already up to date.")
    else:
        print(f"Detected {changed_count} changed or new documents")

        changed_documents_df.createOrReplaceTempView("changed_policy_documents")

        spark.sql(
            f"""
            MERGE INTO {silver_table} AS target
            USING changed_policy_documents AS source
            ON target.record_id = source.record_id
               AND target.is_current = true
            WHEN MATCHED THEN UPDATE SET
              target.valid_to = current_timestamp(),
              target.is_current = false
            """
        )

        final_df = (
            new_chunks_df.alias("chunks")
            .join(changed_documents_df.alias("changed"), on="record_id", how="inner")
            .select("chunks.*")
            .withColumn("processed_at", current_timestamp())
            .withColumn("valid_from", current_timestamp())
            .withColumn("valid_to", lit(None).cast("timestamp"))
            .withColumn("is_current", lit(True))
        )

        final_df.write.format("delta").mode("append").saveAsTable(silver_table)

        print("Silver SCD Type 2 update completed")

# =========================
# JOB SUMMARY
# =========================

current_record_count = spark.table(silver_table).filter(col("is_current") == True).count()
total_record_count = spark.table(silver_table).count()

print(f"Silver processing completed. Target table: {silver_table}")
print(f"Current chunk records: {current_record_count}")
print(f"Total chunk records including history: {total_record_count}")
