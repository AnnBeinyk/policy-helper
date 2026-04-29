# Databricks notebook source

# 05_rag_smoke_test.py

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from databricks.vector_search.client import VectorSearchClient

# =========================
# CONFIG
# =========================

dbutils.widgets.text("embed_endpoint", "databricks-gte-large-en")
dbutils.widgets.text("llm_endpoint", "databricks-gpt-oss-20b")
dbutils.widgets.text("vector_endpoint", "beinyk-vector-endpoint")
dbutils.widgets.text("index_name", "dbr_dev.beinyk_gold.policy_embeddings_index")
dbutils.widgets.text("num_results", "5")

embed_endpoint = dbutils.widgets.get("embed_endpoint")
llm_endpoint = dbutils.widgets.get("llm_endpoint")
vector_endpoint = dbutils.widgets.get("vector_endpoint")
index_name = dbutils.widgets.get("index_name")
num_results = int(dbutils.widgets.get("num_results"))

if num_results <= 0:
    raise ValueError("num_results must be greater than 0")

# =========================
# QUESTIONS
# =========================

questions = [
    "How many paid U.S. holidays does Acme Corp observe each year?",
    "How many vacation weeks does an Acme employee get with less than 10 years of service?",
    "Can an Acme employee carry over unused personal sick time into the next calendar year?",
    "Are supplemental employees eligible for personal choice holidays?",
    "How many paid vacation days do AetherSky Airways pilots accrue annually?",
    "What signing bonus does a Direct-Entry Captain receive at AetherSky Airways?",
    "Are AetherSky pilots paid extra for holiday flight assignments?",
    "What is the domestic per diem rate for AetherSky pilots?",
    "Is an Economy passenger allowed to bring one carry-on bag and one personal item?",
    "What is the checked baggage weight limit for Economy Class?",
    "Can a checked bag over 32 kg be accepted as normal checked baggage?",
    "When must delayed baggage be reported for an international flight?",
    "Should a single transaction of 10,000 USD be flagged under the AML policy?",
    "Should repeated round-number transactions be flagged under the AML policy?",
    "When should a dormant account reactivation be flagged?",
    "When should AML alerts be escalated for compliance review?",
    "What is the minimum credit score for a US loan applicant?",
    "Can a loan applicant with annual income below 30,000 USD qualify under the policy?",
    "What is the maximum debt-to-income ratio allowed for loan approval?",
    "Can a loan amount above 50,000 USD be approved automatically?",
    "What driver license statuses are allowed in the car insurance policy schema?",
    "Does the car insurance policy include vehicle safety inspection as an input parameter?",
    "Is a high-performance vehicle listed as a possible vehicle type?",
    "What output fields does the car insurance compliance policy produce?",
    "Is remote work from another country allowed under the provided policies?",
]

# =========================
# CLIENTS AND HELPERS
# =========================

workspace_client = WorkspaceClient()
vector_client = VectorSearchClient()
index = vector_client.get_index(endpoint_name=vector_endpoint, index_name=index_name)


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


def embed_question(question):
    response = workspace_client.serving_endpoints.query(
        name=embed_endpoint,
        input=[question],
    )
    return extract_embedding(response)


def retrieve_context(query_embedding):
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
        context_blocks.append(
            f"[Source {source_number}]\n"
            f"source_file: {row[1]}\n"
            f"category: {row[2]}\n"
            f"chunk_key: {row[3]}\n"
            f"text: {row[0]}"
        )

    return rows, "\n\n".join(context_blocks)


def answer_question(question, context):
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

    response = workspace_client.serving_endpoints.query(
        name=llm_endpoint,
        messages=[
            ChatMessage(
                role=ChatMessageRole.USER,
                content=prompt,
            )
        ],
    )
    return extract_answer(response)


# =========================
# RUN SMOKE TEST QUESTIONS
# =========================

success_count = 0
failure_count = 0

for question_number, question in enumerate(questions, start=1):
    print("=" * 80)
    print(f"Question {question_number}/{len(questions)}")
    print(question)

    try:
        query_embedding = embed_question(question)
        rows, context = retrieve_context(query_embedding)
        answer = answer_question(question, context)

        print("\nRetrieved sources:")
        for row in rows:
            print(f"- source_file={row[1]}, category={row[2]}, chunk_key={row[3]}")

        print("\nAnswer:")
        print(answer)
        success_count += 1
    except Exception as error:
        print("\nFAILED:")
        print(f"{type(error).__name__}: {error}")
        failure_count += 1

print("=" * 80)
print(f"RAG smoke test completed. Successes: {success_count}. Failures: {failure_count}.")

if failure_count > 0:
    raise ValueError(f"RAG smoke test had {failure_count} failed question(s)")
