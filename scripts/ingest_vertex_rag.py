"""Create a Vertex AI RAG Engine corpus and import the knowledge base into it.

Usage:
    gsutil -m cp kb/docs/*.md gs://YOUR_BUCKET/fixpilot-kb/
    python scripts/ingest_vertex_rag.py --project P --location us-central1 \
        --gcs gs://YOUR_BUCKET/fixpilot-kb/

Prints the corpus resource name; set it as FIXPILOT_RAG_CORPUS.
"""

from __future__ import annotations

import argparse

import vertexai
from vertexai import rag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--gcs", required=True, help="gs:// folder with KB markdown / PDFs")
    ap.add_argument("--display-name", default="fixpilot-kb")
    ap.add_argument("--embedding-model", default="publishers/google/models/text-embedding-005")
    args = ap.parse_args()

    vertexai.init(project=args.project, location=args.location)
    corpus = rag.create_corpus(
        display_name=args.display_name,
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model=args.embedding_model)
            )
        ),
    )
    rag.import_files(
        corpus.name,
        [args.gcs],
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=64)
        ),
    )
    print(corpus.name)


if __name__ == "__main__":
    main()
