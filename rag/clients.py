"""Small HTTP clients with bounded retries and secret-free tracing."""
import json
import time
import threading
import requests
from langsmith import traceable
from rag.config import setting

_local = threading.local()
_embedding_lock = threading.Lock()
_embedding_last = 0.0


def session():
    if not hasattr(_local, 'session'):
        _local.session = requests.Session()
    return _local.session


def post(url, **kwargs):
    for attempt in range(6):
        try:
            r = session().post(url, timeout=180, **kwargs)
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                r.raise_for_status()
                return r.json()
            r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if isinstance(exc, requests.HTTPError) and exc.response.status_code not in (408,429,500,502,503,504):
                raise
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))


@traceable(name='embedding', run_type='embedding')
def embed(text):
    global _embedding_last
    # Gateway quota: 60 requests/minute, burst 20. Pace all worker threads.
    with _embedding_lock:
        delay = 1.05 - (time.monotonic() - _embedding_last)
        if delay > 0: time.sleep(delay)
        _embedding_last = time.monotonic()
        data = post(setting('EMB_URL', 'https://embeddings.parsers360.ru:10443/embed'),
                    headers={'Authorization': 'Bearer ' + setting('EMB_KEY')}, json={'text': text})
    return data['embedding']


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(text)


@traceable(name='financial-reasoning', run_type='llm')
def llm(system, user, max_tokens=5000):
    data = post(setting('LLM_BASE_URL', 'https://claude.gigaparsers.ru').rstrip('/') + '/v1/messages',
                headers={'x-api-key': setting('LLM_API_KEY'), 'anthropic-version': '2023-06-01'},
                json={'model': setting('LLM_MODEL', 'deepseek-flash'), 'temperature': 0,
                      'max_tokens': max_tokens, 'thinking': {'type': 'disabled'},
                      'system': system, 'messages': [{'role': 'user', 'content': user}]})
    text = ''.join(b['text'] for b in data['content'] if b['type'] == 'text')
    if data.get('stop_reason') == 'max_tokens':
        raise RuntimeError('LLM response truncated')
    try:
        return parse_json(text)
    except (ValueError, IndexError):
        raise ValueError('LLM returned invalid JSON; retry the request') from None
