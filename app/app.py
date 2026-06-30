"""
Urdu Fake News Detection System — Hybrid 3-Layer App
Run: streamlit run app.py
"""

import streamlit as st
import pickle, re, os, sys
from scipy.sparse import hstack

sys.path.insert(0, os.path.dirname(__file__))
from hybrid_engine import (extract_core_claim, check_source_credibility,
                           check_entity_credibility, hybrid_verdict)

st.set_page_config(page_title='Urdu Fake News Detector', page_icon='🔍', layout='centered')

# Relative path — works both locally and on Streamlit Cloud
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'classifier.pkl')

def load_model():
    if not os.path.exists(MODEL_PATH): return None
    with open(MODEL_PATH, 'rb') as f: return pickle.load(f)

# Force fresh load every session — no caching
if 'artifacts' not in st.session_state:
    st.session_state.artifacts = load_model()
artifacts = st.session_state.artifacts

def preprocess_urdu(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    text = re.sub(r'[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# Strong institutional/official signals — these appear in real news only
INSTITUTIONAL_SIGNALS = [
    # Named official bodies
    'اسٹیٹ بینک', 'پنجاب فوڈ اتھارٹی', 'وزارت خزانہ', 'وزارت داخلہ',
    'الیکشن کمیشن', 'سپریم کورٹ', 'قومی اسمبلی', 'سینیٹ',
    'پاکستان اسٹاک ایکسچینج', 'ایف بی آر', 'نیب', 'نیپرا',
    'پی ایس ایکس', 'او جی ڈی سی', 'پی ٹی اے',
    # Official actions
    'ترجمان نے بتایا', 'ترجمان کے مطابق', 'پریس کانفرنس میں',
    'سرکاری بیان', 'اعلامیہ جاری', 'نوٹیفکیشن جاری',
    # Specific verifiable numbers
    'فی لیٹر', 'پوائنٹس کی', 'ارب روپے کا', 'کروڑ روپے',
    'فیصد اضافہ', 'فیصد کمی', 'کلو گرام',
    # Named official positions with actions
    'گورنر اسٹیٹ بینک', 'چیئرمین سیکیورٹیز', 'ڈی جی آئی ایس پی آر',
]

def compute_suspicion_score(text):
    """
    Suspicion scorer — runs separately from SVM.
    Counts fake signals, real signals, and institutional signals.
    Returns: (adjustment, fake_hits, real_hits, inst_hits)
    """
    fake_signals = artifacts.get('fake_signals', [])
    real_signals = artifacts.get('real_signals', [])

    fake_hits = [s for s in fake_signals if s in text]
    real_hits = [s for s in real_signals if s in text]
    inst_hits = [s for s in INSTITUTIONAL_SIGNALS if s in text]

    # Each fake signal     = -8% toward real
    # Each real signal     = +5% toward real
    # Each inst signal     = +12% toward real (strong boost for official sources)
    # Inst signals cancel fake signals — if both present, article is real but alarming
    net_fake = max(0, len(fake_hits) - len(inst_hits))  # inst signals neutralize fake signals
    adjustment = (len(real_hits) * 0.05) + (len(inst_hits) * 0.18) - (net_fake * 0.08)
    return adjustment, fake_hits, real_hits, inst_hits

def ml_predict(text):
    clean = preprocess_urdu(text)
    feat  = hstack([artifacts['tfidf_word'].transform([clean]),
                    artifacts['tfidf_char'].transform([clean])])
    pred  = artifacts['model'].predict(feat)[0]
    label = artifacts['label_encoder'].inverse_transform([pred])[0]
    try:    conf = max(artifacts['model'].predict_proba(feat)[0])
    except: conf = 0.65

    # Apply suspicion + institutional adjustment
    adjustment, fake_hits, real_hits, inst_hits = compute_suspicion_score(text)
    base = conf if label == 'real' else (1 - conf)
    adjusted_base = max(0.05, min(0.95, base + adjustment))

    # Convert back to label + conf
    if adjusted_base >= 0.5:
        final_label, final_conf = 'real', adjusted_base
    else:
        final_label, final_conf = 'fake', 1 - adjusted_base

    return final_label, final_conf, fake_hits, real_hits, inst_hits

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center'>🔍 اردو فیک نیوز ڈیٹیکٹر</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray'>Hybrid 3-Layer Urdu Fake News Detection System</p><hr/>", unsafe_allow_html=True)

if artifacts is None:
    st.error("Model not found. Run retrain.py first."); st.stop()
else:
    st.success(f"✅ Model: **{artifacts.get('best_model_name','Classifier')}** | "
               f"Accuracy: **{artifacts.get('test_accuracy',0):.1%}** | "
               f"F1: **{artifacts.get('test_f1',0):.1%}**")

with st.expander("ℹ️ How this 3-layer system works"):
    st.markdown("""
| Layer | What it does | Role |
|-------|-------------|------|
| 🤖 **ML Classifier** | Analyzes writing style & linguistic patterns | **Primary verdict** |
| 🌐 **Claim Verification** | Groq extracts claim → searches news → verifies if it's actually covered | Nudges confidence |
| 👤 **Entity Verification** | Checks if named people actually exist in public records | Nudges confidence |

**Key design:** ML gives the base verdict. Layers 2 & 3 adjust the confidence score up or down — they never override ML alone.
    """)

st.markdown("---")
user_input = st.text_area('📰 Urdu news یہاں paste کریں:', placeholder='خبر یہاں لکھیں...', height=200)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze_btn = st.button('🔍 Analyze News', use_container_width=True, type='primary')
st.markdown("---")

if analyze_btn:
    if not user_input.strip(): st.warning('Please enter some text.'); st.stop()
    if len(preprocess_urdu(user_input)) < 10: st.warning('Text too short.'); st.stop()

    # ── CLAIM EXTRACTION ──────────────────────────────────────────────────────
    with st.status('🧠 Extracting core claim...', expanded=True) as s:
        claim_data = extract_core_claim(user_input)
        st.write(f"**Claim:** {claim_data.get('claim_summary','—')}")
        st.write(f"**Search query:** `{claim_data.get('english_query','—')}`")
        if not claim_data.get('is_physically_possible', True):
            st.write(f"⚠️ **Impossible:** {claim_data.get('impossibility_reason','')}")
        s.update(label='✅ Claim extracted', state='complete', expanded=False)

    # ── LAYER 1: ML + SUSPICION ──────────────────────────────────────────────
    with st.status('🤖 Layer 1 — ML Classifier + Signal Analysis...', expanded=True) as s:
        ml_label, ml_conf, fake_hits, real_hits, inst_hits = ml_predict(user_input)
        st.write(f"ML Prediction: **{ml_label.upper()}** | Confidence: {ml_conf:.1%}")
        if inst_hits:
            st.write(f"🏛️ Institutional signals ({len(inst_hits)}): {', '.join(inst_hits[:3])}")
        if fake_hits:
            st.write(f"⚠️ Fake signals ({len(fake_hits)}): {', '.join(fake_hits[:3])}")
        if real_hits:
            st.write(f"✅ Real signals ({len(real_hits)}): {', '.join(real_hits[:3])}")
        s.update(label=f'✅ Layer 1 — {ml_label.upper()} ({ml_conf:.0%}) | 🏛️{len(inst_hits)} inst | ⚠️{len(fake_hits)} fake', state='complete', expanded=False)

    # ── LAYER 2: SOURCE ───────────────────────────────────────────────────────
    with st.status('🌐 Layer 2 — Verifying claim on news sources...', expanded=True) as s:
        src = check_source_credibility(user_input, claim_data)
        nudge_str = '+20% toward REAL' if src['status']=='confirmed' else \
                    '-35% toward FAKE' if src['status']=='impossible' else \
                    '-15% toward FAKE' if src['status']=='not_confirmed' else \
                    '-10% toward FAKE' if src['status']=='no_results' else 'no change'
        st.write(f"Status: **{src['status'].upper()}** → {nudge_str}")
        st.write(src['label'])
        if src.get('titles'):
            with st.expander('Headlines found'):
                for t in src['titles'][:5]: st.write(f'• {t}')
        s.update(label=f'✅ Layer 2 — {src["status"]} ({nudge_str})', state='complete', expanded=False)

    # ── LAYER 3: ENTITIES ─────────────────────────────────────────────────────
    with st.status('👤 Layer 3 — Verifying named entities...', expanded=True) as s:
        ent = check_entity_credibility(user_input, claim_data)
        nudge_str = '+10% toward REAL' if ent['status']=='verified' else \
                    '-15% toward FAKE' if ent['status']=='unverified' else \
                    '-5% toward FAKE'  if ent['status']=='mixed' else 'no change'
        st.write(f"Status: **{ent['status'].upper()}** → {nudge_str}")
        st.write(ent['label'])
        if ent['verified_persons']:   st.write(f"✅ Verified: {', '.join(ent['verified_persons'])}")
        if ent['unverified_persons']: st.write(f"⚠️ Not found: {', '.join(ent['unverified_persons'])}")
        s.update(label=f'✅ Layer 3 — {ent["status"]} ({nudge_str})', state='complete', expanded=False)

    # ── FINAL VERDICT ─────────────────────────────────────────────────────────
    result = hybrid_verdict(ml_conf, ml_label, src, ent, claim_data)
    b = result['breakdown']

    st.markdown("---")
    st.markdown("### 🏁 Final Verdict")

    if result['verdict'] == 'REAL':
        st.markdown(f"""<div style='background:#e8f5e9;border-left:6px solid #4CAF50;padding:1.4rem;border-radius:8px'>
            <h2 style='color:#2e7d32;margin:0'>✅ REAL NEWS</h2>
            <p style='color:#388e3c;margin:6px 0 0'>Confidence: <b>{result['confidence']:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)
    elif result['verdict'] == 'FAKE':
        st.markdown(f"""<div style='background:#ffebee;border-left:6px solid #F44336;padding:1.4rem;border-radius:8px'>
            <h2 style='color:#c62828;margin:0'>⚠️ FAKE NEWS</h2>
            <p style='color:#d32f2f;margin:6px 0 0'>Confidence: <b>{result['confidence']:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style='background:#fff8e1;border-left:6px solid #FF9800;padding:1.4rem;border-radius:8px'>
            <h2 style='color:#e65100;margin:0'>🟡 UNCERTAIN</h2>
            <p style='color:#bf360c;margin:6px 0 0'>Could not determine with confidence. Verify manually.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### 📊 Score Breakdown")
    st.code(f"""ML base score  : {b['ml_base']:.1f}%  ({ml_label.upper()} at {ml_conf:.0%} confidence)
Source nudge   : {b['src_nudge']:+.1f}%  ({src['status']})
Entity nudge   : {b['ent_nudge']:+.1f}%  ({ent['status']})
─────────────────────────────────────────
Final score    : {b['final']:.1f}%  →  {result['verdict']}""")

    with st.expander("🔬 Full Technical Details"):
        st.markdown("**Claim (Groq)**")
        st.json(claim_data)
        st.markdown("**Layer 2 Headlines**")
        for t in src.get('titles', [])[:5]: st.write(f'• {t}')
        st.markdown("**Layer 3 Entities**")
        st.write(f"Verified: {ent['verified_persons']}")
        st.write(f"Unverified: {ent['unverified_persons']}")

with st.sidebar:
    st.markdown("### 📊 About")
    st.markdown(f"""
**Architecture:** 3-Layer Hybrid AI
**Model:** {artifacts.get('best_model_name','—') if artifacts else '—'}
**Accuracy:** {artifacts.get('test_accuracy',0):.1%}
    """)
    st.markdown("---")
    st.markdown("### 🗂 Datasets")
    st.markdown("""
- [Bend the Truth](https://github.com/MaazAmjad/Datasets-for-Urdu-news)
- [UFN2023](https://doi.org/10.5281/zenodo.7773474)
    """)

