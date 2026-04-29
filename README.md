# Policy Intelligence Assistant

Enterprise-style Databricks RAG project for policy documents. The project has two separate runtime flows:

1. Refresh policy data and embeddings when policies change.
2. Ask questions many times against the already prepared Vector Search index.

This separation is intentional. Policy documents usually change rarely, while users may ask questions often.

## Architecture

1. Raw policy JSON files land in ADLS Gen2 under `raw/Policies/`.
2. Bronze ingests raw JSON with Auto Loader into `dbr_dev.beinyk_bronze.policies_bronze`.
3. Silver cleans and chunks current policy text and maintains SCD Type 2 history in `dbr_dev.beinyk_silver.policy_chunks`.
4. Gold generates embeddings for current chunks into `dbr_dev.beinyk_gold.policy_embeddings`.
5. Databricks Vector Search indexes Gold embeddings as `dbr_dev.beinyk_gold.policy_embeddings_index`.
6. `06_ask_policy.py` embeds a user question, retrieves policy chunks, and calls the LLM endpoint.

## Project Layout

```text
.
|-- databricks.yml
|-- resources/
|   `-- jobs.yml
|-- src/
|   |-- 01_bronze.py
|   |-- 02_silver.py
|   |-- 03_gold.py
|   |-- 04_vector_check.py
|   |-- 05_rag_smoke_test.py
|   `-- 06_ask_policy.py
`-- .github/
    `-- workflows/
        `-- bundle.yml
```

## Runtime Logic

Use `policy_intelligence_assistant_refresh` when policy files are new or changed:

```text
Bronze -> Silver -> Gold -> Vector check
```

Use `policy_intelligence_assistant_ask` or run `src/06_ask_policy.py` directly when you only want to ask questions. This does not rebuild Bronze, Silver, or Gold.

## Tables And Endpoints

Default development configuration:

| Item | Value |
| --- | --- |
| Catalog | `dbr_dev` |
| Bronze schema | `beinyk_bronze` |
| Silver schema | `beinyk_silver` |
| Gold schema | `beinyk_gold` |
| Bronze table | `dbr_dev.beinyk_bronze.policies_bronze` |
| Silver table | `dbr_dev.beinyk_silver.policy_chunks` |
| Gold table | `dbr_dev.beinyk_gold.policy_embeddings` |
| Vector Search index | `dbr_dev.beinyk_gold.policy_embeddings_index` |
| Vector endpoint | `beinyk-vector-endpoint` |
| Embedding endpoint | `databricks-gte-large-en` |
| LLM endpoint | `databricks-gpt-oss-20b` |

## Deploy The Bundle

Validate:

```bash
databricks bundle validate -t dev
```

Deploy:

```bash
databricks bundle deploy -t dev
```

Run a data refresh:

```bash
databricks bundle run policy_intelligence_assistant_refresh -t dev
```

Ask a question from Databricks Jobs UI by opening the deployed `policy-intelligence-assistant-ask-dev` job and overriding the `question` parameter.

You can also open `src/06_ask_policy.py` in Databricks and type into the `question` widget.

## Secrets

The RAG notebook does not need `secret_scope` or `token_secret_key`. It uses the Databricks SDK and the current Databricks execution context to call serving endpoints and Vector Search.

GitHub Actions still needs repository secrets for CI/CD:

| Secret | Purpose |
| --- | --- |
| `DATABRICKS_HOST` | Databricks workspace URL |
| `DATABRICKS_TOKEN` | Token used by GitHub Actions to validate and deploy the bundle |

These CI/CD secrets are stored in GitHub, not in source code.

## CI/CD

Recommended CI/CD flow:

- Pull request to `main`: validate the dev bundle only.
- Push to `main`: validate and deploy the dev bundle. This does not run the refresh job.
- Manual workflow: deploy `dev` or `prod`; optionally run the refresh job after deploy.

This means code changes are deployed automatically, but the expensive policy refresh only runs when you explicitly request it.

In GitHub Actions, create repository secrets:

| Secret | Purpose |
| --- | --- |
| `DATABRICKS_HOST` | Databricks workspace URL |
| `DATABRICKS_TOKEN` | Token used by GitHub Actions to validate and deploy the bundle |

For production, create a GitHub Environment named `prod` and require manual approval before deployment.

## Notes

- Gold embeddings use `WorkspaceClient().serving_endpoints.query(...)` with driver-side batching and `toLocalIterator()`.
- The project avoids `pandas_udf` for embeddings and does not rely on `ai_query` from a standard cluster notebook.
- The Vector Search index should point at the Gold Delta table, with `chunk_key` as the primary key and `embedding` as the vector column.
