"""Store PDF-derived knowledge only, never evaluation answers or labels."""
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(root/'results/knowledge_index.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in ['pages.jsonl','chunks.jsonl','vectors.npy','manifest.json']:
        z.write(root/'index'/name,name)
print('Packaged knowledge index')
