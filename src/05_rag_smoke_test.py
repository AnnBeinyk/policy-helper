# Databricks notebook source

# 05_rag_smoke_test.py

dbutils.widgets.dropdown(
    "demo_question",
    "vacation policy",
    [
        "custom",
        "vacation policy",
        "remote work policy",
        "expense reimbursement",
        "data access policy",
    ],
)
dbutils.widgets.text("question", "")
dbutils.widgets.text("embed_endpoint", "databricks-gte-large-en")
dbutils.widgets.text("llm_endpoint", "databricks-gpt-oss-20b")
dbutils.widgets.text("vector_endpoint", "beinyk-vector-endpoint")
dbutils.widgets.text("index_name", "dbr_dev.beinyk_gold.policy_embeddings_index")
dbutils.widgets.text("num_results", "3")

custom_question = dbutils.widgets.get("question").strip()
demo_question = dbutils.widgets.get("demo_question")

if not custom_question and demo_question == "custom":
    dbutils.widgets.remove("demo_question")
    dbutils.widgets.dropdown(
        "demo_question",
        "vacation policy",
        [
            "custom",
            "vacation policy",
            "remote work policy",
            "expense reimbursement",
            "data access policy",
        ],
    )

# COMMAND ----------

# MAGIC %run ./06_ask_policy

# COMMAND ----------

print("RAG smoke test passed")
