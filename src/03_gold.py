# Databricks notebook source

# 03_gold.py

from databricks.sdk import WorkspaceClient
from pyspark.sql.functions import col, current_timestamp, lit
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# =========================
# CONFIG
# =========================

dbutils.widgets.text("catalog", "dbr_dev")
dbutils.widgets.text("silver_schema", "beinyk_silver")
dbutils.widgets.text("gold_schema", "beinyk_gold")
dbutils.widgets.text("silver_table_name", "policy_chunks")
dbutils.widgets.text("gold_table_name", "policy_embeddings")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en")
dbutils.widgets.text("embedding_batch_size", "16")

catalog = dbutils.widgets.get("catalog")
silver_schema = dbutils.widgets.get("silver_schema")
gold_schema = dbutils.widgets.get("gold_schema")
silver_table_name = dbutils.widgets.get("silver_table_name")
gold_table_name = dbutils.widgets.get("gold_table_name")
embedding_endpoint = dbutils.widgets.get("embedding_endpoint")
embedding_batch_size = int(dbutils.widgets.get("embedding_batch_size"))

if embedding_batch_size <= 0:
    raise ValueError("embedding_batch_size must be greater than 0")

silver_table = f"{catalog}.{silver_schema}.{silver_table_name}"
gold_table = f"{catalog}.{gold_schema}.{gold_table_name}"

# =========================
# CREATE SCHEMA
# =========================

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{gold_schema}")

# =========================
# EMBEDDING HELPERS
# =========================

workspace_client = WorkspaceClient()


def extract_embeddings(response):
    response_dict = response.as_dict() if hasattr(response, "as_dict") else response
    rows = response_dict.get("data") or response_dict.get("predictions") or []

    embeddings = []
    for row in rows:
        if isinstance(row, dict):
            embedding = row.get("embedding") or row.get("embeddings")
        else:
            embedding = getattr(row, "embedding", None)

        if embedding is None:
            raise ValueError(f"Embedding endpoint returned unsupported row format: {row}")

        embeddings.append([float(value) for value in embedding])

    return embeddings


def embed_batch(texts):
    response = workspace_client.serving_endpoints.query(
        name=embedding_endpoint,
        input=texts,
    )
    return extract_embeddings(response)


def iter_batches(rows, batch_size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


# =========================
# LOAD CURRENT SILVER
# =========================

silver_df = (
    spark.table(silver_table)
    .filter(col("is_current") == True)
    .filter(col("chunk").isNotNull())
    .filter(col("chunk_length") > 50)
)

required_columns = [
    "chunk_key",
    "record_id",
    "chunk_index",
    "chunk",
    "chunk_length",
    "category",
    "source_file",
    "path",
    "content_hash",
]
missing_columns = [c for c in required_columns if c not in silver_df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns in Silver table: {missing_columns}")

selected_columns = [
    "chunk_key",
    "record_id",
    "chunk_index",
    "chunk",
    "chunk_length",
    "category",
    "source_file",
    "path",
    "content_hash",
]

# =========================
# INCREMENTAL SELECTION
# =========================

table_exists = spark.catalog.tableExists(gold_table)

if not table_exists:
    print("First run: generating embeddings for all current Silver chunks")
    records_to_refresh_df = silver_df.select("record_id").distinct()
    chunks_to_embed_df = silver_df.select(*selected_columns)
else:
    print("Incremental run: refreshing changed documents and missing current chunks")

    current_silver_documents_df = silver_df.select("record_id", "content_hash").distinct()
    gold_documents_df = spark.table(gold_table).select("record_id", "content_hash").distinct()

    records_to_refresh_df = (
        current_silver_documents_df.alias("s")
        .join(gold_documents_df.alias("g"), on="record_id", how="left")
        .filter(
            col("g.content_hash").isNull()
            | (col("s.content_hash") != col("g.content_hash"))
        )
        .select("s.record_id")
        .distinct()
    )

    chunks_for_refreshed_records_df = (
        silver_df.alias("s")
        .join(records_to_refresh_df.alias("r"), on="record_id", how="inner")
        .select("s.*")
    )

    gold_keys_df = spark.table(gold_table).select("chunk_key").distinct()
    missing_current_chunks_df = (
        silver_df.alias("s")
        .join(gold_keys_df.alias("g"), on="chunk_key", how="left_anti")
        .select("s.*")
    )

    chunks_to_embed_df = (
        chunks_for_refreshed_records_df.select(*selected_columns)
        .unionByName(missing_current_chunks_df.select(*selected_columns))
        .dropDuplicates(["chunk_key"])
    )

chunks_to_embed_count = chunks_to_embed_df.count()

# =========================
# GENERATE EMBEDDINGS
# =========================

if chunks_to_embed_count == 0:
    if not table_exists:
        print("No current Silver chunks found. Gold table was not created.")
        dbutils.notebook.exit("No data to process")

    print("No new or changed chunks found. Gold table is already up to date.")
else:
    print(f"Generating embeddings for {chunks_to_embed_count} chunks")

    embedded_rows = []
    source_rows = chunks_to_embed_df.select(*selected_columns).toLocalIterator()

    for batch in iter_batches(source_rows, embedding_batch_size):
        texts = [row["chunk"] if row["chunk"] is not None else "" for row in batch]
        embeddings = embed_batch(texts)

        if len(embeddings) != len(batch):
            raise ValueError(
                f"Embedding count mismatch. Expected {len(batch)}, got {len(embeddings)}"
            )

        for row, embedding in zip(batch, embeddings):
            embedded_rows.append(
                (
                    row["chunk_key"],
                    row["record_id"],
                    int(row["chunk_index"]),
                    row["chunk"],
                    int(row["chunk_length"]),
                    row["category"],
                    row["source_file"],
                    row["path"],
                    row["content_hash"],
                    embedding,
                )
            )

    embedded_schema = StructType(
        [
            StructField("chunk_key", StringType(), False),
            StructField("record_id", StringType(), False),
            StructField("chunk_index", IntegerType(), False),
            StructField("chunk", StringType(), False),
            StructField("chunk_length", IntegerType(), False),
            StructField("category", StringType(), True),
            StructField("source_file", StringType(), True),
            StructField("path", StringType(), True),
            StructField("content_hash", StringType(), False),
            StructField("embedding", ArrayType(FloatType()), False),
        ]
    )

    embedded_df = (
        spark.createDataFrame(embedded_rows, embedded_schema)
        .withColumn("embedding_model", lit(embedding_endpoint))
        .withColumn("embedded_at", current_timestamp())
    )

    if not table_exists:
        (
            embedded_df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(gold_table)
        )

        spark.sql(
            f"""
            ALTER TABLE {gold_table}
            SET TBLPROPERTIES (
              delta.enableChangeDataFeed = true,
              delta.autoOptimize.optimizeWrite = true,
              delta.autoOptimize.autoCompact = true
            )
            """
        )

        print("Gold table created")
    else:
        records_to_refresh_df.createOrReplaceTempView("changed_gold_record_ids")

        spark.sql(
            f"""
            DELETE FROM {gold_table}
            WHERE record_id IN (
              SELECT record_id FROM changed_gold_record_ids
            )
            """
        )

        embedded_df.write.format("delta").mode("append").saveAsTable(gold_table)

        print("Gold table updated incrementally")

# =========================
# DATA QUALITY CHECKS
# =========================

duplicate_embedding_count = (
    spark.table(gold_table).groupBy("chunk_key").count().filter(col("count") > 1).count()
)

if duplicate_embedding_count > 0:
    raise ValueError(f"Duplicate embeddings detected by chunk_key: {duplicate_embedding_count}")

missing_embedding_count = spark.table(gold_table).filter(col("embedding").isNull()).count()

if missing_embedding_count > 0:
    raise ValueError(f"Rows with missing embeddings detected: {missing_embedding_count}")

# =========================
# JOB SUMMARY
# =========================

gold_record_count = spark.table(gold_table).count()

print(f"Gold processing completed. Target table: {gold_table}")
print(f"Total embedded chunks: {gold_record_count}")
