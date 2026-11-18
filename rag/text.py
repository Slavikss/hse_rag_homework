"""Present generated answers as plain prose without changing financial values."""
import re


def plain_text(text):
    text=re.sub(r'^\s*```[^\n]*\n|^\s*```\s*$', '',text,flags=re.M)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1 (\2)',text)
    text=re.sub(r'\*\*(.+?)\*\*',r'\1',text,flags=re.S)
    text=re.sub(r'__(.+?)__',r'\1',text,flags=re.S)
    text=re.sub(r'`([^`]+)`',r'\1',text)
    text=re.sub(r'(?<!\w)\*([^*\n]+)\*(?!\w)',r'\1',text)
    text=re.sub(r'(?<!\w)_([^_\n]+)_(?!\w)',r'\1',text)
    text=re.sub(r'^[ \t]*>[ \t]+','',text,flags=re.M)
    text=re.sub(r'^[ \t]*#{1,6}[ \t]+','',text,flags=re.M)
    text=re.sub(r'^[ \t]*(?:[-*+] |\d+[.)] )','',text,flags=re.M)
    lines=[]
    for line in text.splitlines():
        if re.fullmatch(r'\s*\|?[\s:|-]+\|\s*',line): continue
        if line.strip().startswith('|') and line.strip().endswith('|'):
            line='; '.join(c.strip() for c in line.strip().strip('|').split('|'))
        lines.append(line)
    return '\n\n'.join(' '.join(p.split()) for p in '\n'.join(lines).split('\n\n') if p.strip())
