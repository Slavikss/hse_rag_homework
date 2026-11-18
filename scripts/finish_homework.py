"""Resume after the quota reset, verify all artifacts, and publish the authorized branch."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from statistics import mean
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
BRANCH='vyacheslav_guch_main'
PYTHON=str(ROOT/'.venv/bin/python')
BASE=ROOT/'results/test_results'
STATUS=ROOT/'outputs/finish_status.json'


def status(stage,**details):
    data={'stage':stage,'updated_at':datetime.now(timezone.utc).isoformat(),**details}
    STATUS.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(json.dumps(data,ensure_ascii=False),flush=True)


def command(args,**kwargs):
    return subprocess.run(args,cwd=ROOT,check=True,**kwargs)


def normalize_report():
    from rag.text import plain_text
    from rag.clients import wait_llm_slot
    from eval.eval_functions import score_answer
    from scripts.run_eval import excel,METRICS
    from langsmith import Client
    rows=[json.loads(x) for x in BASE.with_suffix('.jsonl').read_text().splitlines()]
    assert len(rows)==141
    info=json.loads(BASE.with_suffix('.langsmith.json').read_text())
    client=Client();changed=[]
    (ROOT/'outputs/test_before_plain.jsonl').write_text(BASE.with_suffix('.jsonl').read_text())
    for row in rows:
        answer=plain_text(row['answer']);dirty=answer!=row['answer']
        for step in row['agent_steps']:
            decision=step['decision'];clean=plain_text(decision['answer'])
            dirty=dirty or clean!=decision['answer'];decision['answer']=clean
        if answer!=row['answer']:
            for attempt in range(5):
                try:
                    wait_llm_slot();score=score_answer(row['gold_answer'],answer);break
                except Exception:
                    if attempt==4: raise
                    time.sleep(30)
            row['answer']=answer;row['answer_correctness']=score['answer_correctness']
            row['judge_reason']=plain_text(score['reason'])
            for f in client.list_feedback(run_ids=[row['run_id']],feedback_key=['answer_correctness']):
                client.update_feedback(f.id,score=row['answer_correctness'],comment=row['judge_reason'])
            changed.append(row['question_id'])
        if dirty:
            client.update_run(row['run_id'],outputs={k:row[k] for k in ['answer','context','agent_steps','time']})
    BASE.with_suffix('.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    excel(rows,BASE.with_suffix('.xlsx'))
    summary={k:mean(r[k] for r in rows) for k in METRICS};summary['questions']=len(rows)
    BASE.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
    info['plain_text_presentation']={'note':'Only presentation was changed; financial values and sources were preserved. Changed answers were evaluated again by the unmodified official judge.','rejudged_questions':changed}
    BASE.with_suffix('.langsmith.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    client.flush()
    return summary,info


def update_readme(summary,info):
    path=ROOT/'README.md';text=path.read_text()
    old=next(p for p in text.split('\n\n') if p.startswith('Итоговый прогон'))
    number=lambda x:f'{x:.6f}'.replace('.',',')
    new=('Итоговый прогон выполнен на всех 141 вопросах test. Средний context_recall — '+number(summary['context_recall'])+
         ', context_precision — '+number(summary['context_precision'])+', answer_correctness — '+number(summary['answer_correctness'])+
         '. Среднее время ответа — '+number(summary['time'])+' секунды. Результаты находятся в results/test_results.xlsx. В Excel есть сводка и отдельная строка для каждого вопроса со всеми десятью обязательными полями и объяснением судьи.')
    text=text.replace(old,new).split('\nПубличный эксперимент LangSmith доступен через dataset')[0].rstrip()+'\n'
    text+='\nПубличный эксперимент LangSmith доступен через dataset без авторизации:\n\n'+info['public_dataset']['url']+'\n'
    for url in info.get('public_traces',[]): text+='\nПример трассы:\n\n'+url+'\n'
    text+='\nЧасть прогона была продолжена после сброса суточного лимита LLM. Готовые ответы при восстановлении не пересчитывались. Для нескольких ответов было убрано оформление Markdown без изменения финансовых фактов; такие ответы повторно оценены официальным судьёй.\n'
    path.write_text(text)


def secret_check():
    from dotenv import dotenv_values
    cfg=dotenv_values(ROOT/'.env')
    keys=[cfg.get(k) for k in ['LLM_API_KEY','EMB_KEY','LANGSMITH_API_KEY'] if cfg.get(k)]
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    files={ROOT/n for n in names if n}
    files.update((ROOT/'results').glob('*'))
    for path in files:
        if not path.is_file(): continue
        data=path.read_bytes()
        if any(k.encode() in data for k in keys): raise RuntimeError('Credential found in submission file: '+str(path.relative_to(ROOT)))
    with ZipFile(ROOT/'results/knowledge_index.zip') as archive:
        assert archive.testzip() is None
    assert not subprocess.check_output(['git','ls-files','.env'],cwd=ROOT).strip()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--after',type=float,default=1795046405)
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--detach',action='store_true')
    args=ap.parse_args()
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['LLM_RPM']='10'
    (ROOT/'outputs').mkdir(exist_ok=True)
    if args.detach:
        log=(ROOT/'outputs/auto_complete.log').open('ab',buffering=0)
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--after',str(args.after)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        (ROOT/'outputs/finish.pid').write_text(str(child.pid))
        print('Detached continuation scheduled; process',child.pid,flush=True)
        return
    lock=(ROOT/'outputs/finish.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()==BRANCH
    if args.check:
        assert (ROOT/'index/vectors.npy').exists()
        assert 0<len((ROOT/'results/test_results.jsonl').read_text().splitlines())<=141
        print('Deferred workflow preflight passed; no API calls or publishing performed.')
        return
    status('waiting_for_quota_reset',resume_at=datetime.fromtimestamp(args.after,timezone.utc).isoformat(),pid=os.getpid())
    while time.time()<args.after: time.sleep(max(0,min(55,args.after-time.time())))
    status('evaluating_remaining_questions')
    for attempt in range(3):
        result=subprocess.run([PYTHON,'scripts/run_eval.py','data/test.jsonl','--output','results/test_results.xlsx','--workers','4','--langsmith','--resume'],cwd=ROOT)
        if result.returncode==0: break
        if attempt==2: raise RuntimeError('Evaluation incomplete; inspect outputs/auto_complete.log and saved results')
        time.sleep(55)
    status('normalizing_plain_text_and_verifying')
    summary,info=normalize_report()
    command([PYTHON,'scripts/validate_submission.py'])
    command([PYTHON,'scripts/check_public_experiment.py'])
    update_readme(summary,info)
    secret_check()
    command(['git','diff','--check'])
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()==BRANCH
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode!=0:
        raise RuntimeError('Index already contains staged changes; refusing to commit unrelated work')
    status('committing_and_pushing',metrics=summary)
    command(['git','add','--','README.md','results/knowledge_index.zip','results/test_results.jsonl','results/test_results.xlsx','results/test_results.summary.json','results/test_results.langsmith.json'])
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode!=0:
        command(['git','commit','-m','Add complete test evaluation and public LangSmith experiment'])
    for attempt in range(3):
        pushed=subprocess.run(['git','push','-u','origin',BRANCH],cwd=ROOT)
        if pushed.returncode==0: break
        if attempt==2: raise RuntimeError('Results committed, but push failed; retry git push')
        time.sleep(30)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    status('complete',metrics=summary,commit=revision,branch_url='https://github.com/Slavikss/hse_rag_homework/tree/'+BRANCH,langsmith_url=info['public_dataset']['url'])


if __name__=='__main__':
    try: main()
    except Exception as exc:
        status('failed',error=type(exc).__name__+': '+str(exc))
        raise
