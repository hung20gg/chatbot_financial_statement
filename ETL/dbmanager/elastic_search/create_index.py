import os 
from dotenv import load_dotenv
load_dotenv()

from elasticsearch import Elasticsearch

# Initialize Elasticsearch client
es = Elasticsearch("http://localhost:9200")

def create_index(es, index_name, dims):
    """Create an Elasticsearch index with a dense vector field."""
    mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": dims}
            }
        }
    }

    # Create the index with mapping
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        print(f"Index '{index_name}' created with {dims} dimensions for embeddings.")
    else:
        print(f"Index '{index_name}' already exists.")

# Example usage: Create index with 768-dimensional vectors
create_index(es, "vector_index", dims=768)