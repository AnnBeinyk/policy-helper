# Databricks notebook / 05_rag_test.py

dbutils.widgets.dropdown(
    "demo_question",
    "custom",
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

result = dbutils.notebook.run(
    "./06_ask_policy",
    0,
    {
        "demo_question": dbutils.widgets.get("demo_question"),
        "question": dbutils.widgets.get("question"),
        "embed_endpoint": dbutils.widgets.get("embed_endpoint"),
        "llm_endpoint": dbutils.widgets.get("llm_endpoint"),
        "vector_endpoint": dbutils.widgets.get("vector_endpoint"),
        "index_name": dbutils.widgets.get("index_name"),
        "num_results": dbutils.widgets.get("num_results"),
    },
)

print(result)
print("RAG smoke test passed")
