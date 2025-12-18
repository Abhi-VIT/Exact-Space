# RAG System Design Notes

## Design Trade-offs

### Document Processing

- **Choice**: PyMuPDF for PDF processing
- **Trade-off**: While other options like pdfminer.six offer more control, PyMuPDF provides better performance and ease of use
- **Impact**: Faster document processing but potentially less granular control over PDF parsing

### Chunking Strategy

- **Implementation**: RecursiveCharacterTextSplitter with 500-token chunks and 100-token overlap
- **Trade-off**: Fixed chunk size vs semantic chunking
- **Rationale**:
  - Simpler implementation and predictable memory usage
  - Adequate context preservation with 20% overlap
  - More sophisticated semantic chunking could improve relevance but increase complexity

### Embedding Model

- **Choice**: all-MiniLM-L6-v2 (sentence-transformers)
- **Trade-off**: Model size vs performance
- **Benefits**:
  - Good balance of accuracy and efficiency
  - Well-suited for semantic similarity tasks
  - Smaller footprint compared to larger models

### Vector Database

- **Implementation**: FAISS with IndexFlatL2
- **Trade-off**: Simple L2 distance vs more sophisticated indices
- **Rationale**:
  - Direct implementation of exact nearest neighbor search
  - Suitable for prototype scale
  - Can be upgraded to more sophisticated indices (IVF, HNSW) for scaling

## Retrieval Strategy

### Core Approach

1. **Query Processing**:
   - Direct embedding of user queries using all-MiniLM-L6-v2
   - No query preprocessing or expansion in current implementation

2. **Similarity Search**:
   - L2 distance-based nearest neighbor search
   - Top-k retrieval (default k=5)
   - No re-ranking implemented in prototype

3. **Context Assembly**:
   - Sequential combination of retrieved chunks
   - Source tracking for citation purposes

### Potential Improvements

1. **Hybrid Search**:
   - Combine dense retrieval with sparse (BM25) for better coverage
   - Implement cross-encoder re-ranking

2. **Query Processing**:
   - Add query expansion
   - Implement query type classification

3. **Result Filtering**:
   - Add relevance thresholding
   - Implement semantic deduplication

## Guardrails

### Current Implementation

1. **Source Attribution**:
   - All chunks maintain source information
   - Generated answers include context sources
   - Citation enforcement in prompt structure

2. **Relevance Control**:
   - Top-k limiting of retrieved chunks
   - Direct use of retrieved context in prompt

3. **Error Handling**:
   - Basic document processing error catching
   - Empty result handling

### Recommended Additions

1. **Content Filtering**:
   - Input validation
   - Output content screening
   - Sensitivity detection

2. **Answer Validation**:
   - Source verification
   - Factual consistency checking
   - Confidence scoring

## Scaling Plan

### Document Scale (10x)

1. **Storage Solutions**:
   - Implement disk-based FAISS index
   - Consider distributed vector storage
   - Optimize chunk storage and retrieval

2. **Processing Optimization**:
   - Parallel document processing
   - Batch embedding generation
   - Incremental indexing support

### User Scale (100+ Concurrent)

1. **Architecture Changes**:
   - Implement service-based architecture
   - Add load balancing
   - Deploy caching layers

2. **Performance Optimization**:
   - Query batching
   - Result caching
   - Optimized vector search

### Cost-Efficient Cloud Deployment

1. **Infrastructure**:
   - Use serverless for sporadic workloads
   - Implement auto-scaling
   - Utilize spot instances

2. **Resource Optimization**:
   - Implement tiered storage
   - Use model quantization
   - Deploy efficient serving solutions

## Current Limitations

1. No LLM implementation (placeholder only)
2. Basic error handling
3. Limited query preprocessing
4. No result re-ranking
5. Simple vector search implementation

## Next Steps

1. Implement actual LLM integration
2. Add comprehensive error handling
3. Enhance retrieval with hybrid search
4. Implement result re-ranking
5. Add monitoring and logging
6. Deploy with proper scaling infrastructure