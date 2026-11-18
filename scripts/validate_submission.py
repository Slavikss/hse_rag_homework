"""Audit a completed submission locally; no API calls or new predictions."""
import json
from pathlib import Path
from statistics import mean
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.eval_functions import score_context
from openpyxl import load_workbook

root=Path(__file__).resolve().parents[1]
base=root/'results/test_results'
items=[json.loads(x) for x in (root/'data/test.jsonl').read_text().splitlines()]
rows=[json.loads(x) for x in base.with_suffix('.jsonl').read_text().splitlines()]
refs={r['question_id']:r for r in items}
assert len(rows)==len(items)==141
assert len({r['question_id'] for r in rows})==141
assert {r['question_id'] for r in rows}==set(refs)
metrics=('context_recall','context_precision','answer_correctness','time')
for row in rows:
    ref=refs[row['question_id']]
    for k in ('question','gold_answer','gold_context'): assert row[k]==ref[k],(row['question_id'],k)
    assert row['answer'].strip() and row['judge_reason'].strip()
    assert row['answer_correctness'] in (0,1)
    assert row['time']>0
    for k,v in score_context(row['gold_context'],row['context']).items(): assert abs(row[k]-v)<1e-10
    seen={tuple(p) for step in row['agent_steps'] for p in step['pages']}
    assert all((d['doc_id'],n) in seen for d in row['context'] for n in d['pages'])
summary={k:mean(r[k] for r in rows) for k in metrics}
saved=json.loads(base.with_suffix('.summary.json').read_text())
assert all(abs(summary[k]-saved[k])<1e-10 for k in metrics)
w=load_workbook(base.with_suffix('.xlsx'),read_only=True,data_only=True)
assert w['По вопросам'].max_row==142
for k,c in zip(metrics,next(w['Итоги'].iter_rows(min_row=2,max_row=2))): assert abs(c.value-summary[k])<1e-10
w.close()
print(json.dumps(summary,indent=2))
print('Autonomous follow-up searches:',sum(len(r['agent_steps'])>1 for r in rows))
print('Submission audit passed: all 141 questions, consistent metrics, valid citations.')
