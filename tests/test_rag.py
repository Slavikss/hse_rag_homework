from unittest.mock import patch
from rag.system import RAG
from eval.eval_functions import score_context
from rag.clients import parse_json


def test_agent_actually_searches_again_and_rejects_unseen_citations():
    class Retrieval:
        pages={('bank_2025_annual',1):{'doc_id':'bank_2025_annual','page':1,'text':'first'},
               ('bank_2025_annual',2):{'doc_id':'bank_2025_annual','page':2,'text':'second'}}
        calls=[]
        def search(self,query,docs,k):
            self.calls.append(query)
            return [self.pages[('bank_2025_annual',len(self.calls))]]
    r=RAG.__new__(RAG);r.retriever=Retrieval();r.valid={'bank_2025_annual'}
    r.plan=lambda q:[{'query':'first','doc_ids':['bank_2025_annual']}]
    decisions=[{'answer':'partial','evidence':[],'searches':[{'query':'missing fact','doc_ids':['bank_2025_annual']}]},
               {'answer':'complete','evidence':[{'doc_id':'bank_2025_annual','pages':[2,99]}],'searches':[]}]
    with patch('rag.system.llm',side_effect=decisions): result=r.rag('question')
    assert r.retriever.calls==['first','missing fact']
    assert result['answer']=='complete'
    assert result['context']==[{'doc_id':'bank_2025_annual','pages':[2]}]


def test_score_context_deduplicates_pdf_pages():
    assert score_context([{'doc_id':'d','pages':[1,2]}],[{'doc_id':'d','pages':[2,2]}])=={'context_recall':.5,'context_precision':1.}


def test_markdown_json():
    assert parse_json('```json\n{"answer":"ok"}\n```')=={'answer':'ok'}


def test_preserves_best_numeric_source_when_generator_uses_equivalent_table():
    class Retrieval:
        pages={('report',1):{'doc_id':'report','page':1,'text':'Ending cash 123 456'},
               ('report',2):{'doc_id':'report','page':2,'text':'Balance cash 123 456'}}
        def search(self,*args): return list(self.pages.values())
    r=RAG.__new__(RAG);r.retriever=Retrieval();r.valid={'report'}
    r.plan=lambda q:[{'query':'ending cash','doc_ids':['report']}]
    with patch('rag.system.llm',return_value={'answer':'123 456 thousand rubles','evidence':[{'doc_id':'report','pages':[2]}],'searches':[]}):
        result=r.rag('Ending cash?')
    assert result['context']==[{'doc_id':'report','pages':[1,2]}]
