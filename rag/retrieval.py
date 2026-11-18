import json
import re
import numpy as np
from rank_bm25 import BM25Okapi
from langsmith import traceable
from rag.config import INDEX, ROOT
from rag.clients import embed


def tokens(text):
    return [x[:6] for x in re.findall(r'[а-яёa-z0-9]+', text.lower().replace('ё','е')) if len(x)>2]


class Retriever:
    def __init__(self):
        if not (INDEX/'vectors.npy').exists() and (ROOT/'results/knowledge_index.zip').exists():
            import zipfile
            INDEX.mkdir(exist_ok=True)
            with zipfile.ZipFile(ROOT/'results/knowledge_index.zip') as z: z.extractall(INDEX)
        if not (INDEX/'vectors.npy').exists():
            raise RuntimeError('Build index first: python -m rag.prepare')
        self.chunks=[json.loads(x) for x in (INDEX/'chunks.jsonl').read_text().splitlines()]
        self.pages={(r['doc_id'],r['page']):r for r in map(json.loads,(INDEX/'pages.jsonl').read_text().splitlines())}
        self.vectors=np.load(INDEX/'vectors.npy')
        self.bm25=BM25Okapi([tokens(r['text']) for r in self.chunks])

    @traceable(name='hybrid-search')
    def search(self, query, doc_ids, k=12):
        allowed=set(doc_ids)
        ids=np.array([i for i,r in enumerate(self.chunks) if not allowed or r['doc_id'] in allowed])
        if not len(ids): return []
        v=np.array(embed(query),dtype=np.float32);v/=max(np.linalg.norm(v),1e-12)
        dense=self.vectors[ids]@v
        sparse=self.bm25.get_scores(tokens(query))[ids]
        ranks={}
        for scores in (dense,sparse):
            for rank,j in enumerate(np.argsort(-scores)[:80]):
                i=int(ids[j]);ranks[i]=ranks.get(i,0)+1/(30+rank)
        pages={}
        for i,score in ranks.items():
            r=self.chunks[i];key=(r['doc_id'],r['page'])
            pages[key]=max(pages.get(key,0),score)
        return [self.pages[key] for key in sorted(pages,key=pages.get,reverse=True)[:k]]
