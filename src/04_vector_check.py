# Databricks notebook source

# 04_vector_check.py

from databricks.vector_search.client import VectorSearchClient
import time

# =========================
# CONFIG
# =========================

dbutils.widgets.text("vector_endpoint", "beinyk-vector-endpoint")
dbutils.widgets.text("index_name", "dbr_dev.beinyk_gold.policy_embeddings_index")
dbutils.widgets.text("max_attempts", "20")
dbutils.widgets.text("sleep_seconds", "10")

vector_endpoint = dbutils.widgets.get("vector_endpoint")
index_name = dbutils.widgets.get("index_name")
max_attempts = int(dbutils.widgets.get("max_attempts"))
sleep_seconds = int(dbutils.widgets.get("sleep_seconds"))

if max_attempts <= 0:
    raise ValueError("max_attempts must be greater than 0")

if sleep_seconds <= 0:
    raise ValueError("sleep_seconds must be greater than 0")

# =========================
# CONNECT TO VECTOR SEARCH
# =========================

client = VectorSearchClient()

index = client.get_index(
    endpoint_name=vector_endpoint,
    index_name=index_name,
)

# =========================
# WAIT UNTIL READY
# =========================

print(f"Checking vector index status: {index_name}")

last_status = None

for attempt in range(1, max_attempts + 1):
    description = index.describe()
    status = description.get("status", {})
    last_status = status

    ready = status.get("ready", False)
    state = status.get("state") or status.get("status") or "UNKNOWN"

    print(f"Attempt {attempt}/{max_attempts}: ready={ready}, state={state}")

    if ready:
        print("Vector index is ready")
        break

    if attempt < max_attempts:
        print(f"Waiting {sleep_seconds} seconds...")
        time.sleep(sleep_seconds)

else:
    raise TimeoutError(
        f"Vector index was not ready after {max_attempts} attempts. "
        f"Last status: {last_status}"
    )

# =========================
# SUMMARY
# =========================

description = index.describe()

print("Vector index check completed")
print(f"Endpoint: {vector_endpoint}")
print(f"Index: {index_name}")
print(f"Status: {description.get('status')}")
