"""Run fresh RAG predictions, official scoring, Excel and optional LangSmith experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from statistics import mean
from time import perf_counter
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag.system import RAG
from eval.eval_functions import score_context, score_answer
from eval.evaluator_excel import read_items
from langsmith import Client, tracing_context, trace
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

METRICS=('context_recall','context_precision','answer_correctness','time')
COLUMNS=('question_id','question','gold_context','gold_answer','context','answer',*METRICS)


def excel(rows,path):
    wb=Workbook();s=wb.active;s.title='Итоги'
    s.append(list(METRICS));s.append([mean(r[k] for r in rows) for k in METRICS])
    s.append(['questions',len(rows)])
    ws=wb.create_sheet('По вопросам');ws.append(list(COLUMNS)+['judge_reason'])
    for r in rows:
        values=[r.get(k) for k in (*COLUMNS,'judge_reason')]
        ws.append([json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for v in values])
        for c in ws[ws.max_row]:
            if isinstance(c.value,str): c.data_type='s'
    for sheet in wb:
        sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
        for c in sheet[1]: c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='203864')
        for col in sheet.columns:
            sheet.column_dimensions[col[0].column_letter].width=22
        for row in sheet.iter_rows(min_row=2):
            for c in row: c.alignment=Alignment(vertical='top',wrap_text=True)
    ws.column_dimensions['B'].width=60;ws.column_dimensions['D'].width=65;ws.column_dimensions['F'].width=80
    wb.save(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset',nargs='?',default='data/test.jsonl');ap.add_argument('--output',default='results/test_results.xlsx')
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int)
    ap.add_argument('--langsmith',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    journal=out.with_suffix('.jsonl');items=list(read_items(Path(a.dataset)))
    if a.limit: items=items[:a.limit]
    rows=[]
    if a.resume and journal.exists(): rows=[json.loads(x) for x in journal.read_text().splitlines()]
    done={r['question_id'] for r in rows};system=RAG();client=Client() if a.langsmith else None
    name='financial-rag-'+datetime.now().strftime('%Y%m%d-%H%M%S')
    examples={};project=None
    if client:
        ds=client.create_dataset(dataset_name=name+'-test',description='Financial reports RAG evaluation; references visible only to evaluators')
        ex=[{'id':str(uuid.uuid4()),'inputs':{'question':r['question']},'outputs':{'gold_answer':r['gold_answer'],'gold_context':r['gold_context']},'metadata':{'question_id':r['question_id']}} for r in items]
        client.create_examples(dataset_id=ds.id,examples=ex)
        examples={r['question_id']:e['id'] for r,e in zip(items,ex)}
        project=client.create_project(name,reference_dataset_id=ds.id,description='Hybrid retrieval + autonomous iterative search + official metrics')
        share=client.share_dataset(dataset_id=ds.id)
        Path(out.with_suffix('.langsmith.json')).write_text(json.dumps({'project':name,'project_id':str(project.id),'project_url':project.url,'dataset':str(ds.id),'public_dataset':dict(share)},ensure_ascii=False,default=str,indent=2))
        print('LangSmith',project.url,flush=True)
    def run(item):
        with tracing_context(enabled=bool(client),project_name=name):
            with trace('RAG evaluation',inputs={'question':item['question']},reference_example_id=examples.get(item['question_id'])) as root:
                start=perf_counter();pred=system.rag(item['question']);elapsed=perf_counter()-start
                root.end(outputs={**pred,'time':elapsed})
                run_id=root.id
        # Official judge is left byte-for-byte unchanged; retry transient failures only.
        for attempt in range(5):
            try: judged=score_answer(item['gold_answer'],pred['answer']);break
            except Exception:
                if attempt==4: raise
                import time;time.sleep(2**attempt)
        result={**item,**pred,**score_context(item['gold_context'],pred['context']),
                'answer_correctness':judged['answer_correctness'],'judge_reason':judged['reason'],'time':elapsed,'run_id':str(run_id)}
        if client:
            for k in METRICS: client.create_feedback(run_id,key=k,score=result[k],comment=judged['reason'] if k=='answer_correctness' else None)
        return result
    # Append completed rows immediately; --resume never replays them as fresh predictions.
    with journal.open('a' if a.resume else 'w') as f, ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(run,item):item for item in items if item['question_id'] not in done}
        errors=[]
        for job in as_completed(jobs):
            try: r=job.result()
            except Exception as exc:
                errors.append({'question_id':jobs[job]['question_id'],'error':str(exc)})
                print('FAILED',jobs[job]['question_id'],type(exc).__name__,flush=True)
                continue
            rows.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            print(len(rows),'/',len(items),r['question_id'],*[round(r[k],3) for k in METRICS],flush=True)
    if errors:
        out.with_suffix('.errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
        raise RuntimeError(f'{len(errors)} questions failed; successful predictions saved in {journal}. Use --resume.')
    order={r['question_id']:i for i,r in enumerate(items)};rows.sort(key=lambda r:order[r['question_id']])
    excel(rows,out)
    summary={k:mean(r[k] for r in rows) for k in METRICS};summary['questions']=len(rows)
    out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
    if client:
        client.flush()
        links=[]
        for row in rows[:3]: links.append(client.share_run(row['run_id']))
        path=out.with_suffix('.langsmith.json');info=json.loads(path.read_text());info['public_traces']=links;path.write_text(json.dumps(info,ensure_ascii=False,indent=2))
        client.update_project(project.id,end_time=datetime.now(timezone.utc))

if __name__=='__main__': main()
