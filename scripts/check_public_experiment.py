"""Verify the shared LangSmith dataset and experiment WITHOUT authentication."""
import argparse
import json
from pathlib import Path
import requests

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('metadata',nargs='?',default='results/test_results.langsmith.json')
a=p.parse_args();info=json.loads(Path(a.metadata).read_text())
token=info['public_dataset']['share_token']
base='https://api.smith.langchain.com/public/'+token
r=requests.get(base+'/datasets',timeout=60);r.raise_for_status()
print('Public dataset accessible without API key:',r.json()['name'])
r=requests.get(base+'/datasets/sessions',timeout=60);r.raise_for_status()
projects=r.json()
assert any(str(s['id'])==info['project_id'] for s in projects), 'Experiment not visible through public dataset'
print('Public experiment accessible without API key:',info['project_id'])
