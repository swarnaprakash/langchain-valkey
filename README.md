# LangChain Valkey

LangChain integration for Valkey vector store.

## Installation

```bash
pip install langchain-valkey
```

## Quick Start

```python
from langchain_valkey import ValkeyVectorStore
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
    index_name="my_index",
    embedding=embeddings
)

# Add documents
from langchain_core.documents import Document

docs = [
    Document(page_content="Valkey is a high-performance key-value store", metadata={"source": "doc1"}),
    Document(page_content="Vector databases enable semantic search", metadata={"source": "doc2"}),
]

vectorstore.add_documents(docs)

# Search for similar documents
results = vectorstore.similarity_search("What is Valkey?", k=2)
print(results[0].page_content)
```

## Usage Examples

### Basic Vector Operations

```python
from langchain_valkey import ValkeyVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Initialize
embeddings = OpenAIEmbeddings()
vectorstore = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
    index_name="example_index",
    embedding=embeddings
)

# Add documents
documents = [
    Document(
        page_content="Valkey is an open-source, high-performance key-value datastore.",
        metadata={"category": "database", "type": "key-value"}
    ),
    Document(
        page_content="LangChain is a framework for developing applications with LLMs.",
        metadata={"category": "framework", "type": "ai"}
    )
]

# Add to vector store
ids = vectorstore.add_documents(documents)
print(f"Added documents with IDs: {ids}")

# Similarity search
query = "What is a high-performance database?"
results = vectorstore.similarity_search(query, k=2)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")

# Similarity search with scores
results_with_scores = vectorstore.similarity_search_with_score(query, k=2)
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} - {doc.page_content[:50]}...")

# Get documents by IDs
retrieved_docs = vectorstore.get_by_ids(ids)
print(f"Retrieved {len(retrieved_docs)} documents")

# Delete documents
vectorstore.delete(ids[:1])  # Delete first document
```

### Maximum Marginal Relevance (MMR) Search

MMR search helps diversify results by balancing relevance with diversity, reducing redundant similar documents.

```python
from langchain_valkey import ValkeyVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Initialize
embeddings = OpenAIEmbeddings()
vectorstore = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
    index_name="mmr_example",
    embedding=embeddings
)

# Add diverse documents
documents = [
    Document(page_content="Python is a programming language", metadata={"topic": "python"}),
    Document(page_content="Python programming is easy to learn", metadata={"topic": "python"}),
    Document(page_content="Java is also a programming language", metadata={"topic": "java"}),
    Document(page_content="Machine learning uses Python", metadata={"topic": "ml"}),
    Document(page_content="Web development with Python", metadata={"topic": "web"}),
]

vectorstore.add_documents(documents)

# Regular similarity search - may return similar documents
regular_results = vectorstore.similarity_search("Python programming", k=3)
print("Regular search results:")
for doc in regular_results:
    print(f"- {doc.page_content}")

# MMR search - promotes diversity
mmr_results = vectorstore.max_marginal_relevance_search(
    "Python programming", 
    k=3,
    fetch_k=5,        # Fetch more candidates for diversity selection
    lambda_mult=0.5   # Balance between relevance (1.0) and diversity (0.0)
)
print("\nMMR search results (more diverse):")
for doc in mmr_results:
    print(f"- {doc.page_content}")
```

### Metadata Filtering

```python
# Add documents with rich metadata
documents = [
    Document(
        page_content="Python is a programming language",
        metadata={"language": "python", "category": "programming", "level": "beginner"}
    ),
    Document(
        page_content="Advanced Python concepts include decorators",
        metadata={"language": "python", "category": "programming", "level": "advanced"}
    ),
    Document(
        page_content="JavaScript is used for web development",
        metadata={"language": "javascript", "category": "programming", "level": "beginner"}
    )
]

vectorstore.add_documents(documents)

# Search with metadata filters
results = vectorstore.similarity_search(
    "programming concepts",
    k=5,
    filter={"language": "python", "level": "advanced"}
)
```

### RAG Application with AWS Bedrock

```python
from langchain_valkey import ValkeyVectorStore
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize AWS Bedrock models
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name="us-east-1"
)

llm = BedrockLLM(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1"
)

# Create vector store
vectorstore = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
    index_name="rag_index",
    embedding=embeddings
)

# Add your documents
# vectorstore.add_documents(your_documents)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create RAG chain
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:

Context: {context}

Question: {question}

Answer:
""")

rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Use the chain
answer = rag_chain.invoke("Your question here")
print(answer)
```

## Configuration

LangChain Valkey supports two vector index algorithms with customizable parameters:

### HNSW Index (Default)
The Hierarchical Navigable Small Worlds algorithm provides approximate results with fast query times.

```python
from langchain_valkey import ValkeyVectorStore
from langchain_valkey.config import HNSWConfig, DistanceMetric

# High Performance Configuration
config = HNSWConfig(
    index_name="my_index",
    distance_metric=DistanceMetric.COSINE,
    m=16,                    # Connections per node (default: 16, max: 512)
    ef_construction=200,     # Build quality (default: 200, max: 4096)
    ef_runtime=10           # Search quality (default: 10, max: 4096)
)

vectorstore = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
    index_name="my_index",
    embedding=embeddings,
    config=config
)
```

**Best for:**
- Production workloads
- Performance-critical applications
- Large indexes
- Cases where approximate results are acceptable

### FLAT Index
The Flat algorithm performs brute force linear search, providing exact results.

```python
from langchain_valkey.config import FlatConfig

# Exact Results Configuration
config = FlatConfig(
    index_name="exact_index",
    distance_metric=DistanceMetric.L2,
    vector_dimension=1536
)

vectorstore = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
    index_name="exact_index",
    embedding=embeddings,
    config=config
)
```

**Best for:**
- Development/testing
- Cases requiring exact results
- Small indexes where linear search is practical
- Result validation

### HNSW Parameter Tuning

Three parameters control HNSW performance:

1. **`m`**: Connections per node
   - Higher: Better recall, more memory usage
   - Lower: Less memory, lower recall
   - Range: 1-512, Default: 16

2. **`ef_construction`**: Build quality
   - Higher: Better recall, slower index creation
   - Lower: Faster creation, lower recall
   - Range: 1-4096, Default: 200

3. **`ef_runtime`**: Search quality
   - Higher: Better recall, slower queries
   - Lower: Faster queries, lower recall
   - Range: 1-4096, Default: 10

### Example Configurations

```python
# High Accuracy Configuration
high_accuracy_config = HNSWConfig(
    index_name="high_accuracy",
    distance_metric=DistanceMetric.COSINE,
    m=32,                    # More connections for better recall
    ef_construction=400,     # Higher build quality
    ef_runtime=40           # Better search quality
)

# Fast Queries Configuration
fast_config = HNSWConfig(
    index_name="fast_queries",
    distance_metric=DistanceMetric.COSINE,
    m=8,                     # Fewer connections for speed
    ef_construction=40,      # Faster build
    ef_runtime=4            # Faster queries
)
```

### Distance Metrics

Supported distance metrics for vector similarity:

- **COSINE**: Cosine similarity (default) - measures angle between vectors
- **L2**: Euclidean distance - measures straight-line distance
- **IP**: Inner product - measures dot product similarity

### Additional Resources

For detailed information on index parameters, limits, and advanced configuration options:
- [Valkey Search FT.CREATE Documentation](https://github.com/valkey-io/valkey-search/blob/main/COMMANDS.md#ftcreate)
- [Valkey Search Documentation](https://valkey.io/topics/search/) for performance optimization strategies

## Configuration Options

### Index Types

- **HNSW**: Hierarchical Navigable Small World - optimized for speed
- **FLAT**: Brute force search - optimized for accuracy

### Distance Metrics

- **COSINE**: Cosine similarity (default)
- **L2**: Euclidean distance
- **IP**: Inner product

### Connection Options

```python
# Basic connection
vectorstore = ValkeyVectorStore(
    valkey_url="valkey://localhost:6379",
    index_name="my_index",
    embedding=embeddings
)

# With authentication
vectorstore = ValkeyVectorStore(
    valkey_url="valkey://username:password@localhost:6379",
    index_name="my_index",
    embedding=embeddings
)

# With SSL
vectorstore = ValkeyVectorStore(
    valkey_url="valkeyss://localhost:6380",
    index_name="my_index",
    embedding=embeddings
)
```

## Testing

The project includes comprehensive test coverage organized into three categories:

### Unit Tests
```bash
# Unit tests (algorithms and business logic)
poetry run pytest tests/unit_tests/ --cov=langchain_valkey --cov-branch
```

### Integration Tests
```bash
# Integration tests (core functionality with live Valkey server)
# Requires: Valkey server with search module loaded running on localhost:6379
poetry run pytest tests/integration_tests/ --cov=langchain_valkey --cov-branch
```

### LangChain Standard Tests
```bash
# LangChain interface compliance tests
# Ensures ValkeyVectorStore behaves consistently with other LangChain vector stores
poetry run pytest tests/integration_tests/test_langchain_standard.py
```

### All Tests
```bash
# Complete test suite
poetry run pytest tests/
```
