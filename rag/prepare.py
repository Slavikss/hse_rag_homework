"""Download, parse/OCR and embed the PDF corpus. No QA labels are read."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile
import numpy as np
import pymupdf
import requests
from rag.config import ROOT, INDEX
from rag.clients import embed

REPORT_URL = 'https://media.githubusercontent.com/media/polIINNa/hse_rag_homework/main/data/reports.zip'


def ensure_reports():
    docs = [json.loads(x) for x in (ROOT/'data/documents.jsonl').read_text().splitlines()]
    base = ROOT/'data/reports'
    if not all((base/d['path']).is_file() for d in docs):
        archive = ROOT/'data/reports.zip'
        if not zipfile.is_zipfile(archive):
            with requests.get(REPORT_URL, stream=True, timeout=180) as r:
                r.raise_for_status()
                with archive.open('wb') as out:
                    for block in r.iter_content(1024*1024): out.write(block)
        with zipfile.ZipFile(archive) as z:
            z.extractall(base)
    return docs


def ocr(page):
    """TSV word geometry retains row/column relations, including table numbers."""
    from collections import defaultdict
    import csv
    import io
    binary = os.getenv('TESSERACT_CMD', 'tesseract')
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp)/'page.png'
        page.get_pixmap(dpi=240).save(image)
        result = subprocess.run([binary,str(image),'stdout','-l','rus+eng','--psm','3','tsv'],
                                text=True,capture_output=True,check=True)
    lines = defaultdict(list)
    for w in csv.DictReader(io.StringIO(result.stdout), delimiter='\t', quoting=csv.QUOTE_NONE):
        if w.get('text','').strip() and float(w['conf']) >= 0:
            y = int(w['top']) + int(w['height'])/2
            # Merge independently detected blocks at the same table row.
            key = round(y/12)
            lines[key].append((int(w['left']),w['text']))
    return '\n'.join('  '.join(t for _,t in sorted(words)) for _,words in sorted(lines.items()))


def parse_doc(meta):
    target = INDEX/'parsed'/f"{meta['doc_id']}.jsonl"
    if target.exists():
        return [json.loads(x) for x in target.read_text().splitlines()]
    rows=[]
    with pymupdf.open(ROOT/'data/reports'/meta['path']) as doc:
        if len(doc) != meta['pages']:
            raise ValueError(f"Page count mismatch: {meta['doc_id']}")
        for i,p in enumerate(doc):
            text=p.get_text(sort=True)
            method='text'
            if len(text.strip()) < 100:
                text=ocr(p)
                method='ocr'
            rows.append({'doc_id':meta['doc_id'],'page':i+1,'organization':meta['organization'],
                         'text':text,'method':method})
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print('parsed',meta['doc_id'],len(rows),'OCR',sum(r['method']=='ocr' for r in rows),flush=True)
    return rows


def chunks(pages):
    rows=[]
    for page in pages:
        text=page['text']
        for start in range(0, max(1,len(text)), 2300):
            part=text[start:start+2800]
            if part.strip():
                rows.append({'doc_id':page['doc_id'],'page':page['page'],
                             'text':f"{page['organization']} | {page['doc_id']} | PDF page {page['page']}\n"+part})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--parse-only',action='store_true')
    args=parser.parse_args()
    INDEX.mkdir(exist_ok=True)
    (INDEX/'parsed').mkdir(exist_ok=True)
    docs=ensure_reports()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pages=[row for group in pool.map(parse_doc,docs) for row in group]
    (INDEX/'pages.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pages))
    print('Pages',len(pages),'OCR',sum(r['method']=='ocr' for r in pages),flush=True)
    if args.parse_only: return
    parts=chunks(pages)
    (INDEX/'chunks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in parts))
    cache=INDEX/'embeddings'
    cache.mkdir(exist_ok=True)
    def compute(row):
        key=hashlib.sha256(row['text'].encode()).hexdigest()
        path=cache/f'{key}.npy'
        if path.exists(): return np.load(path)
        arr=np.asarray(embed(row['text']),dtype=np.float32)
        np.save(path,arr)
        return arr
    vectors=[None]*len(parts)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(compute,row):i for i,row in enumerate(parts)}
        errors=[]
        for n,future in enumerate(as_completed(jobs),1):
            try: vectors[jobs[future]]=future.result()
            except Exception as exc:
                errors.append(str(exc));print('EMBED ERROR',type(exc).__name__,str(exc)[:150],flush=True)
            if n%100==0: print('embedded',n,'/',len(parts),flush=True)
    if errors: raise RuntimeError(f'{len(errors)} embeddings failed; rerun to resume cached vectors')
    matrix=np.asarray(vectors,dtype=np.float32)
    matrix/=np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-12)
    np.save(INDEX/'vectors.npy',matrix)
    manifest={'pages':len(pages),'chunks':len(parts),'dimension':matrix.shape[1],
              'documents_sha256':hashlib.sha256((ROOT/'data/documents.jsonl').read_bytes()).hexdigest(),
              'parser':'pymupdf-layout+tsv-ocr-240dpi','chunk_size':2800,'stride':2300}
    (INDEX/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(manifest)


if __name__=='__main__': main()
