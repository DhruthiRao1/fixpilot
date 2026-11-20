# Deploying FixPilot on Google Cloud

Two deployment targets share the same code:

| Target | What runs | When to use |
|---|---|---|
| **Cloud Run** (`Dockerfile`, `cloudbuild.yaml`) | FastAPI service wrapping the LangGraph workflow | Integrating with a web/mobile app, Teams/Slack bot, or ITSM portal |
| **Vertex AI Agent Engine** (`adk deploy agent_engine`) | The ADK agent (`src/fixpilot/adk_agent`) | Managed agent runtime with sessions, tracing and Agent-to-Agent composition |

## 1. One-time project setup

```bash
PROJECT=your-project; REGION=us-central1
gcloud config set project $PROJECT
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com aiplatform.googleapis.com secretmanager.googleapis.com

gcloud artifacts repositories create fixpilot --repository-format=docker --location=$REGION

# Least-privilege runtime identity
gcloud iam service-accounts create fixpilot-runtime
gcloud projects add-iam-policy-binding $PROJECT \
  --member=serviceAccount:fixpilot-runtime@$PROJECT.iam.gserviceaccount.com \
  --role=roles/aiplatform.user
```

## 2. Knowledge base (Vertex AI RAG Engine)

```bash
gsutil mb -l $REGION gs://$PROJECT-fixpilot-kb
gsutil -m cp kb/docs/*.md gs://$PROJECT-fixpilot-kb/
python scripts/ingest_vertex_rag.py --project $PROJECT --location $REGION \
  --gcs gs://$PROJECT-fixpilot-kb/
# store the printed corpus name as a secret
echo -n "projects/.../ragCorpora/123" | gcloud secrets create fixpilot-rag-corpus --data-file=-
gcloud secrets add-iam-policy-binding fixpilot-rag-corpus \
  --member=serviceAccount:fixpilot-runtime@$PROJECT.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

## 3a. Cloud Run via Cloud Build

```bash
gcloud builds submit --config cloudbuild.yaml --substitutions=SHORT_SHA=$(git rev-parse --short HEAD)
```

The pipeline runs unit tests and the offline eval gate before building, so a regression in
diagnosis accuracy or a safety violation blocks the deploy.

## 3b. Vertex AI Agent Engine via ADK

```bash
pip install -e ".[adk]"
cd src/fixpilot
adk deploy agent_engine --project=$PROJECT --region=$REGION \
  --display_name="FixPilot" --trace_to_cloud adk_agent
```

## Production notes

* **Session state:** the default `InMemorySaver` is per-instance. For multiple Cloud Run
  instances, swap in a shared LangGraph checkpointer (e.g. Postgres on Cloud SQL / AlloyDB)
  in `build_graph(checkpointer=...)`.
* **Auth:** the service is deployed `--no-allow-unauthenticated`; front it with IAP or API
  Gateway for employee SSO.
* **PII:** images are never persisted; on-screen text is redacted before state, logs and
  tickets.
