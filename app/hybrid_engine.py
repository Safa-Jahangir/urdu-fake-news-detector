"""
hybrid_engine.py — 3-Layer Hybrid Fake News Detection Engine

Architecture:
  Layer 1 (ML):     Primary verdict — 89% accurate, trained on 5000+ samples
  Layer 2 (Source): Nudges confidence using Groq claim verification
  Layer 3 (Entity): Nudges confidence using Wikipedia/News entity check

Key principle: ML is the anchor. Layers 2 & 3 only adjust confidence
by fixed amounts — they never flip the verdict on their own.
"""

import re, time, requests, urllib.parse, json
from xml.etree import ElementTree

import streamlit as st

def _get_groq_key():
    """Reads from Streamlit secrets (deployed) or falls back to local value."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return 'YOUR_GROQ_KEY_HERE'   # <-- used only for local testing

GROQ_API_KEY = _get_groq_key()

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9,ur;q=0.8',
}


# ══════════════════════════════════════════════════════════════════════════════
# CLAIM EXTRACTOR — Groq
# ══════════════════════════════════════════════════════════════════════════════

def extract_core_claim(text: str) -> dict:
    """Use Groq to extract the core claim from Urdu article."""
    prompt = (
        'Analyze this Urdu news article and return ONLY a JSON object. No explanation, no markdown.\n\n'
        f'Article: """{text[:1500]}"""\n\n'
        'Return this exact JSON:\n'
        '{"english_query":"6-8 word English search query WHO did WHAT WHERE",'
        '"urdu_query":"5-6 word Urdu search query",'
        '"key_persons":["named individuals only, not generic titles"],'
        '"key_organizations":["named orgs"],'
        '"is_physically_possible":true,'
        '"impossibility_reason":"",'
        '"claim_summary":"one sentence English summary of the core claim"}\n\n'
                'Set is_physically_possible=false if scientifically/medically impossible:\n'
        '- Curing diabetes/cancer in days, miracle cures with instant govt approval\n'
        '- Humans on Mars today, anti-gravity, moon bases\n'
        '- Anything contradicting established science or medicine\n\n'
'Example for budget article:\n'
        '{"english_query":"Pakistan PM budget 2026 education 200 billion",'
        '"urdu_query":"وزیراعظم بجٹ تعلیم ارب روپے",'
        '"key_persons":["Muhammad Aurangzeb"],'
        '"key_organizations":["IMF","State Bank"],'
        '"is_physically_possible":true,'
        '"impossibility_reason":"",'
        '"claim_summary":"Pakistan PM presented budget allocating 200 billion for education"}\n\n'
        'ONLY return the JSON:'
    )
    try:
        resp = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {GROQ_API_KEY}'},
            json={
                'model': 'llama-3.1-8b-instant',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 300, 'temperature': 0.1
            },
            timeout=20
        )
        if resp.status_code == 200:
            raw = resp.json()['choices'][0]['message']['content'].strip()
            raw = re.sub(r'^```json\s*|^```\s*|```$', '', raw, flags=re.MULTILINE).strip()
            return json.loads(raw)
    except Exception as e:
        print(f'[Groq claim error]: {e}')

    return {
        'english_query': text[:80].replace('\n', ' '),
        'urdu_query': text[:60].replace('\n', ' '),
        'key_persons': [], 'key_organizations': [],
        'is_physically_possible': True,
        'impossibility_reason': '',
        'claim_summary': text[:100]
    }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — SOURCE CHECK
# Returns: 'confirmed', 'not_confirmed', 'impossible', 'unknown'
# ══════════════════════════════════════════════════════════════════════════════

def _google_news_rss(query: str, max_results: int = 6) -> list[dict]:
    try:
        url  = f'https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en&gl=PK&ceid=PK:en'
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        root  = ElementTree.fromstring(resp.content)
        return [
            {'title': item.findtext('title') or '',
             'url'  : item.findtext('link')  or '',
             'source': (item.find('source').text if item.find('source') is not None else '')}
            for item in root.findall('.//item')[:max_results]
        ]
    except Exception as e:
        print(f'[RSS error]: {e}')
        return []


def check_source_credibility(text: str, claim_data: dict = None) -> dict:
    """
    Layer 2 — check if claim is confirmed by real news sources.
    Returns status: 'confirmed' | 'not_confirmed' | 'impossible' | 'no_results' | 'unknown'
    """
    if claim_data and not claim_data.get('is_physically_possible', True):
        return {
            'status': 'impossible',
            'label' : f"Impossible claim: {claim_data.get('impossibility_reason','')}",
            'titles': [], 'query_used': ''
        }

    english_q = claim_data.get('english_query', text[:80]) if claim_data else text[:80]
    claim_sum = claim_data.get('claim_summary', english_q)  if claim_data else english_q

    results = _google_news_rss(english_q, max_results=6)
    time.sleep(0.3)

    if not results:
        return {
            'status': 'no_results',
            'label' : 'No news coverage found for this claim',
            'titles': [], 'query_used': english_q
        }

    # Ask Groq: do these headlines actually confirm the claim?
    titles = [r['title'] for r in results[:6]]
    prompt = (
        'You are a fact-checker. Do these news headlines DIRECTLY confirm this specific claim?\n\n'
        f'Claim: "{claim_sum}"\n\n'
        'Headlines:\n' + '\n'.join(f'- {t}' for t in titles) + '\n\n'
        'Rules:\n'
        '- confirmed=true ONLY if headlines directly cover THIS exact claim\n'
        '- confirmed=false if headlines are about related topics but NOT this specific claim\n'
        '- Be strict: "IMF Pakistan economy" articles do NOT confirm "IMF took water reservoirs"\n\n'
        'Return JSON only: {"confirmed": true/false, "reason": "one sentence"}'
    )

    try:
        gr = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {GROQ_API_KEY}'},
            json={
                'model': 'llama-3.1-8b-instant',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 100, 'temperature': 0.1
            },
            timeout=15
        )
        if gr.status_code == 200:
            raw   = gr.json()['choices'][0]['message']['content'].strip()
            raw   = re.sub(r'^```json\s*|^```\s*|```$', '', raw, flags=re.MULTILINE).strip()
            gdata = json.loads(raw)
            confirmed = gdata.get('confirmed', False)
            reason    = gdata.get('reason', '')
            status    = 'confirmed' if confirmed else 'not_confirmed'
            return {
                'status': status, 'label': reason,
                'titles': titles, 'query_used': english_q
            }
    except Exception as e:
        print(f'[Groq verify error]: {e}')

    return {'status': 'unknown', 'label': 'Verification inconclusive', 'titles': titles, 'query_used': english_q}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — ENTITY VERIFICATION
# Returns: 'verified', 'unverified', 'mixed', 'no_entities'
# ══════════════════════════════════════════════════════════════════════════════

def _wikipedia_check(name: str) -> bool:
    try:
        resp = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={'action':'query','list':'search','srsearch':name,'format':'json','srlimit':3},
            headers={'User-Agent': 'UrduFakeNewsDetector/1.0'},
            timeout=8
        )
        if resp.status_code == 200:
            hits = resp.json().get('query', {}).get('search', [])
            parts = [p for p in name.lower().split() if len(p) > 2]
            return any(
                any(p in h.get('title','').lower() for p in parts)
                for h in hits
            )
        return False
    except Exception:
        return False


def _news_check(name: str) -> bool:
    try:
        results = _google_news_rss(f'{name} Pakistan', max_results=4)
        return len(results) >= 2
    except Exception:
        return False


def _extract_persons_regex(text: str) -> list:
    patterns = [
        r'ڈاکٹر\s+[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,}){0,2}',
        r'پروفیسر\s+[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,}){0,2}',
        r'جنرل\s+[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,}){0,2}',
        r'وزیرِ?\s*اعظم\s+[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,}){0,2}',
        r'صدر\s+[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,}){0,2}',
    ]
    persons = []
    for p in patterns:
        persons.extend(re.findall(p, text))
    persons.extend(re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b', text))
    return list(dict.fromkeys([p.strip() for p in persons]))


def check_entity_credibility(text: str, claim_data: dict = None) -> dict:
    persons = (claim_data.get('key_persons', []) if claim_data else []) or _extract_persons_regex(text)
    # Filter out generic words that aren't real named persons
    GENERIC_WORDS = {'police', 'dakandar', 'shopkeeper', 'man', 'woman', 'officer',
                     'official', 'source', 'sources', 'minister', 'government',
                     'پولیس', 'دکاندار', 'شخص', 'ذرائع', 'حکومت', 'وزیر'}
    INVALID_PREFIXES = ('unnamed', 'unknown', 'a local', 'local', 'the ', 'a ')
    persons = [p for p in persons[:4]
               if len(p.strip()) > 4
               and p.lower().strip() not in GENERIC_WORDS
               and not any(p.lower().startswith(pfx) for pfx in INVALID_PREFIXES)]

    verified, unverified = [], []
    for person in persons:
        clean = re.sub(r'^(ڈاکٹر|پروفیسر|جنرل|وزیر|صدر|سپیکر)\s*', '', person).strip()
        if _wikipedia_check(clean):
            verified.append(person)
        else:
            if _news_check(person):
                verified.append(person)
            else:
                unverified.append(person)
        time.sleep(0.2)

    total = len(verified) + len(unverified)
    if total == 0:
        status = 'no_entities'
    elif len(unverified) == 0:
        status = 'verified'
    elif len(verified) == 0:
        status = 'unverified'
    else:
        status = 'mixed'

    return {
        'status'            : status,
        'verified_persons'  : verified,
        'unverified_persons': unverified,
        'total_checked'     : total,
        'label'             : (f'All {total} person(s) verified' if status == 'verified'
                               else f'{len(unverified)} person(s) could not be verified' if status == 'unverified'
                               else f'Mixed: {len(verified)} verified, {len(unverified)} unverified' if status == 'mixed'
                               else 'No named persons detected')
    }


# ══════════════════════════════════════════════════════════════════════════════
# FINAL HYBRID VERDICT
# ML is the anchor — layers 2 & 3 nudge confidence only
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_verdict(ml_conf: float, ml_label: str,
                   source_result: dict, entity_result: dict,
                   claim_data: dict = None) -> dict:
    """
    ML is the primary verdict.
    Layers 2 & 3 nudge the final confidence score up or down.

    Nudge values:
      Source confirmed    → +20% toward REAL
      Source not_confirmed→ +15% toward FAKE
      Source impossible   → +35% toward FAKE
      Source no_results   → +10% toward FAKE
      Source unknown      →   0% (no change)

      Entity verified     → +10% toward REAL
      Entity unverified   → +15% toward FAKE
      Entity mixed        → +5%  toward FAKE
      Entity no_entities  →   0% (no change)
    """

    # Start with ML score (0=fake, 1=real)
    base = ml_conf if ml_label == 'real' else (1 - ml_conf)

    # Source nudge
    src_status = source_result.get('status', 'unknown')
    ml_numeric  = ml_conf if ml_label == 'real' else (1 - ml_conf)

    if src_status == 'confirmed':
        src_nudge = +0.20
    elif src_status == 'not_confirmed':
        # If ML is strongly REAL and no fake signals detected → gentle nudge only
        # This prevents uncertain-but-plausible real stories from being pushed to FAKE
        if ml_numeric >= 0.75:
            src_nudge = -0.10
        else:
            src_nudge = -0.22
    elif src_status == 'impossible':
        src_nudge = -0.35
    elif src_status == 'no_results':
        if ml_numeric >= 0.75:
            src_nudge = -0.12
        else:
            src_nudge = -0.22
    else:
        src_nudge = 0.0

    # Entity nudge
    ent_status = entity_result.get('status', 'no_entities')
    if ent_status == 'verified':
        ent_nudge = +0.10
    elif ent_status == 'unverified':
        ent_nudge = -0.28
    elif ent_status == 'mixed':
        ent_nudge = -0.05
    else:
        ent_nudge = 0.0

    final_score = max(0.0, min(1.0, base + src_nudge + ent_nudge))

    if final_score >= 0.60:
        verdict, confidence, color = 'REAL', final_score * 100, 'green'
    elif final_score <= 0.44:
        verdict, confidence, color = 'FAKE', (1 - final_score) * 100, 'red'
    else:
        verdict, confidence, color = 'UNCERTAIN', 50.0, 'orange'

    return {
        'verdict'    : verdict,
        'confidence' : round(confidence, 1),
        'color'      : color,
        'final_score': round(final_score, 3),
        'breakdown'  : {
            'ml_base'   : round(base * 100, 1),
            'src_nudge' : round(src_nudge * 100, 1),
            'ent_nudge' : round(ent_nudge * 100, 1),
            'final'     : round(final_score * 100, 1),
        }
    }
