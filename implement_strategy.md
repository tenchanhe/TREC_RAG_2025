# TREC 2025 RAG Track Overview

The TREC (Text REtrieval Conference) RAG (Retrieval-Augmented Generation) Track is a leading international evaluation for systems combining large-scale retrieval and generative models. The 2025 edition continues the tradition, using the MS MARCO V2.1 corpus and offering multiple tasks to foster advances in retrieval and generation technologies.

---

## 📂 Data Components

### 1. Corpus

* **Official corpus:** MS MARCO Segment v2.1 collection

  * **Document Corpus:**
    * Total Documents: 10,960,555
    * Format: 70 gzipped JSONL files in TAR archive
    * Download: msmarco_v2.1_doc.tar (28.1 GB)
    * MD5: a5950665d6448d3dbaf7135645f1e074
    * Document Structure:
      * `docid`: Unique identifier (format: msmarco_v2.1_doc_[filename]_[byte_offset])
      * `url`: Source URL
      * `title`: Document title
      * `headings`: Subheadings within the document
      * `body`: Main content of the document

  * **Segmented Corpus:**
    * Total Segments: 113,520,750
    * Format: 70 gzipped JSONL files in TAR archive
    * Download: msmarco_v2.1_doc_segmented.tar (25.1 GB)
    * MD5: 3799e7611efffd8daeb257e9ccca4d60
    * Segmentation Method: Sliding window of 10 sentences with 5-sentence stride
    * Segment Size: ~500-1000 characters
    * Segment Structure:
      * `docid`: Segment identifier
      * `url`: Source URL
      * `title`: Document title
      * `headings`: Subheadings within the document
      * `segment`: Segment text
      * `start_char`: Start character offset in original document
      * `end_char`: End character offset in original document

  * **Download:** See the [TREC RAG official corpus announcement page](https://trec-rag.github.io/annoucements/2025-rag25-corpus/)

### 2. Qrel (Relevance Judgments)

* **Content:** Human-annotated relevance between each topic (query) and corpus document/segment
* **Format example:**

  ```
  topic-id    0    docid    relevance
  ```

  * `relevance` is typically 0 (not relevant) or 1 (relevant); sometimes finer-grained
* **Purpose:** Used as ground truth for evaluating retrieval and generation results
* **Download:** Provided on the official announcement page

### 3. Topics (Queries)

* **Content:** Each entry is a natural language query or information need
* **Format:** Usually a `topics.rag25.*.txt` file, each line contains a topic id and the query text
* **Purpose:** Used as input for retrieval and generation tasks
* **Current status:** As of early June 2025, the official evaluation topics (queries) have **not yet been released**. Please monitor the [official announcements](https://trec-rag.github.io/annoucements/2025-rag25-corpus/) for updates.

---

## 🧪 Task Overview

We are conducting three tasks in TREC 2025 RAG track. These tasks are as follows:

| Task Name                          | Description                                                                                                                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **(R) Retrieval Task**             | Requires participants to rank and retrieve the most relevant segments from the MS MARCO Segment v2.1 collection based on a given set of input topics (queries).                                              |
| **(AG) Augmented Generation Task** | Requires participants to generate RAG answers, including attributions to supporting segments from the MS MARCO Segment v2.1 collection, using top-k segments from a baseline retriever.                      |
| **(RAG) Retrieval-Augmented Task** | Requires participants to generate RAG answers with supporting citations, using custom retrieval systems and segmentation strategies. Segments must be mapped to the official MS MARCO corpus for evaluation. |

---

## 🧾 Evaluation with `trec_eval`

We will use the `trec_eval` tool to evaluate retrieval results:

```bash
$ trec_eval -m map -m ndcg_cut.10 path/to/qrels.txt path/to/run.txt
```

Ensure that run files conform to TREC format:

```
topic-id Q0 docid rank score STANDARD
```

---

## 📁 Project Structure

```
project_root/
├── data/                      # Data directory
│   ├── corpus/               # MS MARCO V2.1 corpus files
│   │   ├── raw/             # Raw downloaded files
│   │   ├── processed/       # Processed corpus files
│   │   └── validation/      # Validation results
│   ├── qrels/               # Relevance judgments
│   └── topics/              # Topic files (queries)
│
├── indexes/                  # Search indices
│   ├── bm25/                # BM25 indices
│   ├── dense/               # Dense indices
│   └── hybrid/              # Hybrid indices
│
├── runs/                     # Retrieval and generation results
│   ├── retrieval/           # Retrieval run files
│   ├── generation/          # Generation outputs
│   └── evaluation/          # Evaluation results
│
├── src/                      # Source code
│   ├── data/                # Data processing modules
│   ├── retrieval/           # Retrieval modules
│   ├── generation/          # Generation modules
│   ├── evaluation/          # Evaluation modules
│   └── utils/               # Utility functions
│
├── configs/                  # Configuration files
│   ├── data_config.yaml     # Data processing config
│   ├── retrieval_config.yaml # Retrieval config
│   └── generation_config.yaml # Generation config
│
├── tests/                    # Test files
│   ├── test_data/           # Data processing tests
│   ├── test_retrieval/      # Retrieval tests
│   └── test_generation/     # Generation tests
│
├── logs/                     # Log files
│   ├── data_processing/     # Data processing logs
│   ├── retrieval/           # Retrieval logs
│   └── generation/          # Generation logs
│
├── notebooks/                # Jupyter notebooks
│   ├── data_analysis/       # Data analysis notebooks
│   ├── experiments/         # Experiment notebooks
│   └── visualization/       # Visualization notebooks
│
├── docs/                     # Documentation
│   ├── api/                 # API documentation
│   ├── examples/            # Usage examples
│   └── reports/             # Project reports
│
├── scripts/                  # Utility scripts
│   ├── setup.sh             # Setup script
│   ├── download.sh          # Download script
│   └── evaluate.sh          # Evaluation script
│
├── .env                      # Environment variables
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup file
├── README.md                # Project documentation
└── LICENSE                  # License file
```

### Directory Descriptions

#### 1. Data Directory (`data/`)
* **corpus/**: Contains all corpus-related files
  * `raw/`: Original downloaded files
  * `processed/`: Preprocessed corpus files
  * `validation/`: Validation results and reports
* **qrels/**: Relevance judgment files
* **topics/**: Query/topic files

#### 2. Indexes Directory (`indexes/`)
* **bm25/**: BM25 sparse indices
* **dense/**: Dense embedding indices
* **hybrid/**: Combined indices

#### 3. Runs Directory (`runs/`)
* **retrieval/**: TREC format run files
* **generation/**: Generated answers
* **evaluation/**: Evaluation results

#### 4. Source Code (`src/`)
* **data/**: Data processing modules
* **retrieval/**: Retrieval system modules
* **generation/**: Generation system modules
* **evaluation/**: Evaluation modules
* **utils/**: Utility functions

#### 5. Configuration (`configs/`)
* YAML configuration files for different components

#### 6. Tests (`tests/`)
* Unit tests and integration tests

#### 7. Logs (`logs/`)
* Processing and execution logs

#### 8. Notebooks (`notebooks/`)
* Jupyter notebooks for analysis and experiments

#### 9. Documentation (`docs/`)
* API documentation and usage examples

#### 10. Scripts (`scripts/`)
* Utility scripts for common tasks

### File Naming Conventions

1. **Python Files**
   * Use snake_case: `data_processor.py`
   * Test files: `test_data_processor.py`

2. **Configuration Files**
   * Use snake_case with .yaml: `data_config.yaml`

3. **Log Files**
   * Include timestamp: `data_processing_20240315.log`

4. **Run Files**
   * Include run ID: `retrieval_run_001.txt`

5. **Notebook Files**
   * Use descriptive names: `corpus_analysis.ipynb`

---

## ⚙️ Technical Strategy: Hybrid Retrieval + Parent-Child RAG

We will adopt the following approach to improve retrieval-augmented generation:

### ✅ Retrieval Strategy

**Hybrid Retrieval (Step-by-Step):**

1. **Sparse Index Construction**

   * Preprocess the corpus and convert to Anserini-compatible format
   * Use Pyserini to build a BM25 index:

     ```bash
     python -m pyserini.index.lucene \
       --collection JsonCollection \
       --input data/corpus/ \
       --index indexes/bm25 \
       --generator DefaultLuceneDocumentGenerator \
       --threads 8 \
       --storePositions --storeDocvectors --storeRaw
     ```

2. **Dense Embedding Generation**

   * Use HuggingFace models like DPR or Contriever to embed the segments
   * Store embeddings in FAISS or Annoy index

3. **Query Encoding and Retrieval**

   * Encode queries with same model
   * Perform top-k search over both BM25 and dense index

4. **Fusion Strategy**

   * Combine sparse and dense results using Reciprocal Rank Fusion (RRF) or similar methods

### 🧩 Parent-Child Document Mapping

Due to the sliding-window segmentation:

* Answer generation may span multiple segments
* We will build a `segment-to-document` index using `docid`, `segment_start_char`, and `segment_end_char` fields from the corpus

This mapping enables grouping of segments for context expansion.

### 🔄 Answer Generation Pipeline

1. **Hybrid Retrieval**

   * Use fused ranking to select top-k segments per query

2. **Context Construction**

   * For each selected segment, retrieve sibling segments from the same parent document
   * Assemble them into a single context block

3. **LLM-Based Generation**

   * Feed constructed context into a generative model (e.g., T5, Mistral)
   * Ensure answers contain in-text segment citations by `docid` to facilitate evaluation

## 📝 Additional Implementation Details

### 1. Data Preparation and Validation
* **Data Validation Scripts**
  * Verify MS MARCO V2.1 corpus integrity
  * Validate JSONL format and required fields
  * Check segment boundaries and document mappings

* **Data Preprocessing Pipeline**
  * JSONL format conversion utilities
  * Text cleaning and normalization
  * Segment boundary detection and adjustment

* **Data Analysis Tools**
  * Corpus statistics generation
  * Distribution analysis of segment lengths
  * Document and segment relationship visualization

### 2. Extended Evaluation Metrics
* **Retrieval Metrics**
  * MAP (Mean Average Precision)
  * NDCG@10 (Normalized Discounted Cumulative Gain)
  * MRR (Mean Reciprocal Rank)
  * Precision@k

* **Generation Metrics**
  * ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
  * BLEU score
  * Citation accuracy
  * Factual consistency
  * Answer relevance

### 3. Technical Components
* **Caching System**
  * Embedding cache for frequently accessed segments
  * Query result cache
  * Model output cache

* **Logging System**
  * Experiment tracking
  * Performance monitoring
  * Error logging
  * Resource usage tracking

* **Configuration Management**
  * Experiment configuration files
  * Model hyperparameters
  * System settings
  * Environment variables

* **Checkpoint Management**
  * Model state saving
  * Training progress tracking
  * Experiment state preservation

### 4. Experiment Management
* **Tracking System**
  * Retrieval strategy comparisons
  * Generation model performance
  * Hyperparameter optimization results
  * Error analysis and case studies

### 5. Enhanced Project Structure
```
project-root/
├── src/
│   ├── retrieval/      # Retrieval-related code
│   ├── generation/     # Generation-related code
│   ├── evaluation/     # Evaluation-related code
│   └── utils/         # Utility functions
├── configs/           # Configuration files
├── experiments/       # Experiment results
└── tests/            # Unit tests
```

### 6. Environment Setup
* **Documentation**
  * Detailed environment setup guide
  * Dependency management (requirements.txt/pyproject.toml)
  * Docker configuration (if needed)
  * System requirements

### 7. Documentation
* **Technical Documentation**
  * API documentation
  * Usage examples
  * FAQ
  * Contributing guidelines

### 8. Testing Strategy
* **Test Coverage**
  * Unit tests
  * Integration tests
  * Performance tests
  * End-to-end tests

### 9. Development Workflow
* **Version Control**
  * Git workflow
  * Branch management
  * Code review process

* **CI/CD Pipeline**
  * Automated testing
  * Code quality checks
  * Documentation updates
  * Deployment automation

## 🚀 Detailed Implementation Steps

### Track R: Retrieval Task Implementation

#### 1. Data Processing Scripts
```bash
src/data/
├── download_corpus.py          # Download and verify corpus files
├── extract_corpus.py          # Extract TAR and gzip files
├── validate_corpus.py         # Validate corpus integrity
├── preprocess_documents.py    # Document preprocessing
└── preprocess_segments.py     # Segment preprocessing
```

#### 2. Index Construction Scripts
```bash
src/retrieval/
├── build_bm25_index.py        # Build BM25 index using Pyserini
├── build_dense_index.py       # Build dense embeddings index
├── build_hybrid_index.py      # Combine sparse and dense indices
└── index_utils.py            # Index utility functions
```

#### 3. Retrieval Pipeline Scripts
```bash
src/retrieval/
├── query_processor.py         # Query preprocessing
├── sparse_retriever.py       # BM25 retrieval
├── dense_retriever.py        # Dense retrieval
├── hybrid_retriever.py       # Hybrid retrieval with RRF
└── retrieval_utils.py        # Retrieval utility functions
```

#### 4. Evaluation Scripts
```bash
src/evaluation/
├── prepare_runs.py           # Prepare TREC format run files
├── evaluate_retrieval.py     # Run trec_eval
├── analyze_results.py        # Analyze retrieval results
└── visualization.py          # Result visualization
```

### Track AG: Augmented Generation Task Implementation

#### 1. Context Processing Scripts
```bash
src/generation/
├── context_processor.py      # Process retrieved segments
├── context_assembler.py      # Assemble context for generation
└── citation_extractor.py     # Extract citations from context
```

#### 2. Generation Pipeline Scripts
```bash
src/generation/
├── model_loader.py           # Load generation models
├── answer_generator.py       # Generate answers
├── citation_generator.py     # Generate citations
└── post_processor.py         # Post-process generated text
```

#### 3. Evaluation Scripts
```bash
src/evaluation/
├── evaluate_generation.py    # Evaluate generation quality
├── evaluate_citations.py     # Evaluate citation accuracy
├── rouge_evaluator.py       # ROUGE score calculation
└── bleu_evaluator.py        # BLEU score calculation
```

### Track RAG: Retrieval-Augmented Task Implementation

#### 1. Custom Retrieval Scripts
```bash
src/rag/
├── custom_segmenter.py       # Custom segmentation strategy
├── custom_retriever.py       # Custom retrieval system
└── segment_mapper.py         # Map to official corpus
```

#### 2. RAG Pipeline Scripts
```bash
src/rag/
├── rag_processor.py          # RAG pipeline processor
├── context_builder.py        # Build context for RAG
├── answer_generator.py       # Generate RAG answers
└── citation_manager.py       # Manage citations
```

#### 3. Evaluation Scripts
```bash
src/evaluation/
├── evaluate_rag.py           # Evaluate RAG performance
├── evaluate_factuality.py    # Evaluate factual consistency
└── evaluate_attribution.py   # Evaluate attribution quality
```

### Common Infrastructure

#### 1. Configuration Management
```bash
configs/
├── data_config.yaml          # Data processing configuration
├── retrieval_config.yaml     # Retrieval configuration
├── generation_config.yaml    # Generation configuration
└── evaluation_config.yaml    # Evaluation configuration
```

#### 2. Logging and Monitoring
```bash
src/utils/
├── logger.py                 # Logging setup
├── metrics_tracker.py        # Track experiment metrics
└── resource_monitor.py       # Monitor system resources
```

#### 3. Testing Framework
```bash
tests/
├── test_data/               # Test data
├── test_retrieval/          # Retrieval tests
├── test_generation/         # Generation tests
└── test_rag/               # RAG tests
```

### Implementation Timeline

1. **Phase 1: Data Processing (Week 1-2)**
   * Set up data processing pipeline
   * Implement corpus validation
   * Create preprocessing scripts

2. **Phase 2: Retrieval System (Week 3-4)**
   * Implement BM25 index
   * Implement dense retrieval
   * Develop hybrid retrieval

3. **Phase 3: Generation System (Week 5-6)**
   * Set up generation pipeline
   * Implement citation system
   * Develop post-processing

4. **Phase 4: RAG Integration (Week 7-8)**
   * Integrate custom retrieval
   * Implement RAG pipeline
   * Develop evaluation system

5. **Phase 5: Optimization (Week 9-10)**
   * Performance optimization
   * Error analysis
   * System tuning

6. **Phase 6: Final Testing (Week 11-12)**
   * End-to-end testing
   * Performance evaluation
   * Documentation completion

## 📝 實作範例：Vicarious Trauma 查詢流程

讓我們以一個具體的例子來說明整個 RAG 系統的工作流程：

### 1. 查詢輸入 (Topics)
在 `topics.rag24.test.txt` 中，我們有以下查詢：
```
2024-145979    what is vicarious trauma and how can it be coped with?
```

### 2. 相關性判斷 (Qrels)
在 `qrels.rag24.test-umbrela-all.txt` 中，我們可以看到相關性評分：
```
2024-145979 Q0 msmarco_v2.1_doc_13_1647729865#1_3617399591 3
```
這表示該文檔片段與查詢的相關性評分為 3 分。

### 3. 文檔內容 (Corpus)
使用以下指令可以查看相關文檔內容：
```bash
zcat msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_13.json.gz | grep 'doc_13_1647729865' | sed -n '2p' | jq '.segment'
```

文檔內容：
```
Often, people who help behind the scenes in helping those with trauma might neglect their own exposure. In this article, you will learn what Vicarious Trauma is, some of the symptoms associated with it. You will also learn how to cope with Vicarious Trauma so that you can live a happy and balanced professional life. What is Vicarious Trauma & Who is Susceptible to it? Vicarious Trauma refers to the trauma you may experience when being indirectly exposed to a traumatic event. It could be a situation where the life of another person was being threatened or witnessing the death of someone else. It is as if you've taken this trauma as your own, and it has triggered a shocking ...
```

這個例子展示了：
1. 查詢如何從 topics 文件開始
2. 如何通過 qrels 找到相關文檔
3. 如何從 corpus 中檢索實際文檔內容
4. 文檔內容如何與查詢相關
