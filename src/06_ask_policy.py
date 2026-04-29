# Databricks notebook source

# 06_ask_policy.py

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient

# =========================
# CONFIG
# =========================

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

demo_question = dbutils.widgets.get("demo_question")
custom_question = dbutils.widgets.get("question").strip()
embed_endpoint = dbutils.widgets.get("embed_endpoint")
llm_endpoint = dbutils.widgets.get("llm_endpoint")
vector_endpoint = dbutils.widgets.get("vector_endpoint")
index_name = dbutils.widgets.get("index_name")
num_results = int(dbutils.widgets.get("num_results"))

demo_questions = {
    "vacation policy": "How many vacation weeks do employees get?",
    "remote work policy": "Is working remotely from another country allowed?",
    "expense reimbursement": "Can an employee expense a business dinner?",
    "data access policy": "Who is allowed to access confidential customer data?",
}

question = custom_question or demo_questions.get(demo_question)

if not question:
    question = demo_questions["vacation policy"]

if num_results <= 0:
    raise ValueError("num_results must be greater than 0")

# =========================
# SERVING ENDPOINT HELPERS
# =========================

workspace_client = WorkspaceClient()


def as_dict(response):
    return response.as_dict() if hasattr(response, "as_dict") else response


def extract_embedding(response):
    response_dict = as_dict(response)
    rows = response_dict.get("data") or response_dict.get("predictions") or []

    if not rows:
        raise ValueError(f"Embedding endpoint returned no rows: {response_dict}")

    first_row = rows[0]

    if isinstance(first_row, dict):
        embedding = first_row.get("embedding") or first_row.get("embeddings")
    else:
        embedding = getattr(first_row, "embedding", None)

    if not embedding:
        raise ValueError(f"Embedding endpoint returned an empty embedding: {response_dict}")

    return [float(value) for value in embedding]


def extract_answer(response):
    response_dict = as_dict(response)

    if response_dict.get("choices"):
        answer_content = response_dict["choices"][0]["message"]["content"]
        if isinstance(answer_content, list):
            return "".join(
                part.get("text", "")
                for part in answer_content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return answer_content

    predictions = response_dict.get("predictions")
    if predictions:
        first_prediction = predictions[0]
        if isinstance(first_prediction, str):
            return first_prediction
        if isinstance(first_prediction, dict):
            return (
                first_prediction.get("content")
                or first_prediction.get("text")
                or first_prediction.get("generated_text")
            )

    raise ValueError(f"LLM endpoint returned unsupported response format: {response_dict}")


def query_llm(prompt):
    messages = [{"role": "user", "content": prompt}]

    return workspace_client.serving_endpoints.query(
        name=llm_endpoint,
        inputs={"messages": messages},
    )


# =========================
# EMBED QUESTION
# =========================

embedding_response = workspace_client.serving_endpoints.query(
    name=embed_endpoint,
    input=[question],
)
query_embedding = extract_embedding(embedding_response)

# =========================
# RETRIEVE CONTEXT
# =========================

vector_client = VectorSearchClient()
index = vector_client.get_index(endpoint_name=vector_endpoint, index_name=index_name)

results = as_dict(
    index.similarity_search(
        query_vector=query_embedding,
        columns=["chunk", "source_file", "category", "chunk_key"],
        num_results=num_results,
    )
)

rows = results.get("result", {}).get("data_array", [])

if not rows:
    raise ValueError("No results returned from vector search")

context_blocks = []

for source_number, row in enumerate(rows, start=1):
    chunk = row[0]
    source_file = row[1]
    category = row[2]
    chunk_key = row[3]

    context_blocks.append(
        f"[Source {source_number}]\n"
        f"source_file: {source_file}\n"
        f"category: {category}\n"
        f"chunk_key: {chunk_key}\n"
        f"text: {chunk}"
    )

context = "\n\n".join(context_blocks)

# =========================
# GENERATE ANSWER
# =========================

prompt = f"""
You are a policy intelligence assistant.

Answer the user's question using ONLY the provided policy context.
If the answer is not present in the context, say: "Not found in the provided policies."

For compliance-style questions, give the decision first: Allowed, Not allowed, Conditional, or Not found.
Mention the relevant source_file if possible.

Policy context:
{context}

Question:
{question}
"""

answer_response = query_llm(prompt)
answer_text = extract_answer(answer_response)

if not answer_text:
    raise ValueError(f"LLM returned an empty answer: {as_dict(answer_response)}")

# =========================
# OUTPUT
# =========================

print("Question:")
print(question)

print("\nRetrieved sources:")
for row in rows:
    print(f"- source_file={row[1]}, category={row[2]}, chunk_key={row[3]}")

print("\nAnswer:")
print(answer_text)
