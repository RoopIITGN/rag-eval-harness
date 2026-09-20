"""Create three Azure AI Search indexes, one per chunking configuration.

Design notes:

- exhaustiveKnn, not HNSW. At ~200 chunks a brute-force scan is well under a
  millisecond, so an approximate index would cost recall for a latency saving
  that doesn't exist. See README.

- doc_id, char_start, char_end are retrievable because gold labels are
  character spans in the source document. Without them, eval can't tell whether
  a retrieved chunk overlaps the answer.

- The content field uses the English analyzer so BM25 gets stemming and
  stopword handling on prose, which is most of the corpus.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, SearchFieldDataType,
    VectorSearch, VectorSearchProfile, ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters, VectorSearchAlgorithmMetric,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CONFIGS = ["256-fixed", "512-fixed", "512-recursive"]
DIMS = 1536          # text-embedding-3-small


def build(name: str) -> SearchIndex:
    return SearchIndex(
        name=f"circulars-{name}",
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="doc_id", type=SearchFieldDataType.String,
                        filterable=True, facetable=True),
            SearchableField(name="content", type=SearchFieldDataType.String,
                            analyzer_name="en.microsoft"),
            SimpleField(name="char_start", type=SearchFieldDataType.Int32),
            SimpleField(name="char_end", type=SearchFieldDataType.Int32),
            SimpleField(name="token_count", type=SearchFieldDataType.Int32),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=DIMS,
                vector_search_profile_name="exhaustive-profile",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[ExhaustiveKnnAlgorithmConfiguration(
                name="exhaustive-config",
                parameters=ExhaustiveKnnParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE),
            )],
            profiles=[VectorSearchProfile(
                name="exhaustive-profile",
                algorithm_configuration_name="exhaustive-config",
            )],
        ),
    )


def main():
    client = SearchIndexClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
    )
    for cfg in CONFIGS:
        idx = build(cfg)
        client.create_or_update_index(idx)
        print(f"created/updated index: {idx.name}")

    existing = [i for i in client.list_index_names()]
    print(f"\nindexes on service: {existing}")


if __name__ == "__main__":
    main()
