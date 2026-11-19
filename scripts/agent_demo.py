"""Show autonomous follow-up search with a deliberately small first-page budget.

This separate experiment is not part of the 141-question test evaluation.
Only the first retrieval pass is limited to one page per planned query.
The LLM itself decides whether more retrieval is necessary.
"""
import json
from pathlib import Path
import sys
import threading
from datetime import datetime
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag.agentic import AgenticRAG
from langsmith import Client,tracing_context,trace


def main():
    client=Client();system=AgenticRAG()
    remaining=0;lock=threading.Lock()
    original_plan=system.plan;original_search=system.retriever.search
    def plan(question):
        nonlocal remaining
        searches=original_plan(question)
        remaining=min(5,len(searches))
        return searches
    def search(*args,**kwargs):
        nonlocal remaining
        with lock:
            limited=remaining>0
            if limited: remaining-=1
        pages=original_search(*args,**kwargs)
        return pages[:1] if limited else pages
    system.plan=plan;system.retriever.search=search
    question='Сопоставь чистую прибыль или убыток за полный 2024 год у девяти организаций: Сбербанк, ВТБ, Т-Банк, Банк ДОМ.РФ, Московский кредитный банк, Россельхозбанк, Совкомбанк, Альфа-Банк (Беларусь) и НПФ Газпромбанк-фонд. Для каждой организации укажи собственную валюту и единицы отчёта, значение и страницы источника. Не подменяй группу отдельным банком и год полугодием. Если данных не хватает, найди их дополнительным поиском.'
    project='financial-rag-budget-demo-'+datetime.now().strftime('%Y%m%d-%H%M%S')
    with tracing_context(enabled=True,project_name=project,client=client):
        with trace('Дополнительный поиск при малой начальной выборке',inputs={'question':question,'initial_page_budget_per_query':1}) as root:
            start=perf_counter();result=system.rag(question);elapsed=perf_counter()-start
            root.end(outputs={**result,'time':elapsed})
    client.flush();url=client.share_run(root.id)
    path=Path(__file__).resolve().parents[1]/'results/agentic_demo.json'
    path.write_text(json.dumps({'question':question,'initial_page_budget_per_query':1,'part_of_test_evaluation':False,**result,'time':elapsed,'trace_url':url},ensure_ascii=False,indent=2))
    print('Agent turns',len(result['agent_steps']),flush=True)
    print('Public trace',url,flush=True)


if __name__=='__main__': main()
