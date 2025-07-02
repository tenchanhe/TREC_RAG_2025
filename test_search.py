from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import LuceneHnswDenseSearcher

searcher = LuceneHnswDenseSearcher('./indexes/test/embeddings.jsonl')
hits = searcher.search('Random Correlated Normal Variables')

for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')