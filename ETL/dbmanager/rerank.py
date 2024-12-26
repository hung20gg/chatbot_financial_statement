from huggingface_hub import AsyncInferenceClient
import asyncio
from FlagEmbedding import FlagReranker

from pydantic import BaseModel, SkipValidation, ConfigDict
from typing import Any, Union
import numpy as np
import json


class BaseRerannk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name : str
    reranker : Union[FlagReranker, AsyncInferenceClient] = None
    reranker_type : str = None
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        self._init_reranker()
        
    def _init_reranker(self):
        if 'http' in self.name:
            self.reranker = AsyncInferenceClient(self.name)
            self.reranker_type = 'api'
        else:
            self.reranker = FlagReranker(self.name)
            self.reranker_type = 'local'
     
     
    # from https://github.com/plaggy/rag-containers/blob/main/rag-container-lance/rag-gradio-async/backend/semantic_search.py    
    async def _rerank_api(self, query, documents, top_k, batch_size = 32, **kwargs):
        scores = []
        for i in range(int(np.ceil(len(documents) / batch_size))):
            resp = await self.reranker.post(
                json={
                    "query": query,
                    "texts": documents[i * batch_size:(i + 1) * batch_size],
                    "truncate": True
                }
            )
            try:
                batch_scores = json.loads(resp)
                batch_scores = [s["score"] for s in batch_scores]
                scores.extend(batch_scores)
            except Exception as e:
                print(e)
                print(resp)
                scores.extend([0] * len(documents[i * batch_size:(i + 1) * batch_size]))
            
        documents = [doc for _, doc in sorted(zip(scores, documents))[-top_k:]] 
        return documents 
        
        
    def _rerank_local(self, query, documents, top_k, **kwargs):
        
        pairs = [[query, doc] for doc in documents]
        
        scores = self.reranker.compute_score(pairs, **kwargs)
        
        best_docs = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in best_docs]
            
    def rerank(self, query, documents, top_k, **kwargs):
        
        if self.reranker_type == 'api':
            result = asyncio.run(self._rerank_api(query, documents, top_k, **kwargs))
        else:
            result = self._rerank_local(query, documents, top_k, **kwargs)
            
        return result
    
    def rerank_langchain(self, query, documents, top_k, **kwargs):
        mapping_documents = {doc.page_content:doc for doc in documents}
        
        mapping_keys = list(mapping_documents.keys())
        
        selected_docs = self.rerank(query, mapping_keys, top_k, **kwargs)   
        
        return [mapping_documents[doc] for doc in selected_docs]
        
        
    