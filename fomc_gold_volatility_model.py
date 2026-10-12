"""
Quantitative NLP: Predicting Gold (XAU/USD) Volatility Regimes from FOMC Press Conferences
==========================================================================================
Methodology:
- Speaker Diarization: Isolating Chair Powell's statements from journalist inquiries
- Market Realization: Ground-truth alignment with COMEX Gold Futures (GC=F) intraday ranges
- Monetary Feature Engineering: N-Gram (1-3) TF-IDF with monetary-specific stopword handling
- Similarity-Weighted Probability: Cosine distance matching against historical meeting vectors
- Statistical Validation: Out-of-sample (2025-2026) classification report, ROC-AUC, Brier score
"""

import sys
import os
import re
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss, accuracy_score

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def extract_powell_speech(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    pattern = r'<NAME>(.*?)</NAME>[\.\:\s]*(.*?)(?=<NAME>|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    powell_dialogues = []
    for speaker, speech in matches:
        if 'POWELL' in speaker.strip().upper():
            c = re.sub(r'\s+', ' ', speech).strip()
            if c:
                powell_dialogues.append(c)
    return " ".join(powell_dialogues)


def load_all_fomc_transcripts(folder_path='dataset'):
    records = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt') and 'FOMC' in f])
    for filename in files:
        m = re.search(r'(\d{8})', filename)
        if not m:
            continue
        dt = pd.to_datetime(m.group(1), format='%Y%m%d')
        speech = extract_powell_speech(os.path.join(folder_path, filename))
        if len(speech) > 200:
            records.append({
                'filename': filename,
                'date': dt,
                'date_str': dt.strftime('%Y-%m-%d'),
                'word_count': len(speech.split()),
                'speech_text': speech
            })
    df = pd.DataFrame(records).sort_values(by='date').reset_index(drop=True)
    return df


def fetch_and_merge_gold_volatility(df_transcripts, cache_csv='gold_prices_cache.csv'):
    if os.path.exists(cache_csv):
        gold = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
    else:
        min_date = (df_transcripts['date'].min() - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        max_date = (df_transcripts['date'].max() + pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        gold = yf.download('GC=F', start=min_date, end=max_date, progress=False)
        if isinstance(gold.columns, pd.MultiIndex):
            gold.columns = gold.columns.get_level_values(0)
        gold.to_csv(cache_csv)

    pips_day, pips_ext, closes = [], [], []
    for _, row in df_transcripts.iterrows():
        sub = gold.loc[gold.index >= row['date']].head(2)
        if len(sub) > 0:
            pips_day.append(round(float(sub.iloc[0]['High'] - sub.iloc[0]['Low']) * 10, 1))
            pips_ext.append(round(float(sub['High'].max() - sub['Low'].min()) * 10, 1))
            closes.append(round(float(sub.iloc[0]['Close']), 2))
        else:
            pips_day.append(np.nan)
            pips_ext.append(np.nan)
            closes.append(np.nan)

    df_transcripts['gold_close'] = closes
    df_transcripts['pips_same_day'] = pips_day
    df_transcripts['pips_extended'] = pips_ext
    return df_transcripts.dropna(subset=['pips_same_day']).reset_index(drop=True)


class FOMCGoldVolatilityModel:
    def __init__(self, pip_threshold=400, top_k=5, ngram_range=(1, 3)):
        self.pip_threshold = pip_threshold
        self.top_k = top_k
        self.ngram_range = ngram_range

        base_stops = set(ENGLISH_STOP_WORDS)
        fomc_chatter = {'chair', 'powell', 'thank', 'thanks', 'good', 'afternoon', 'morning', 'today', 'question', 'think', 'know', 'say', 'said', 'going'}
        preserve = {'not', 'no', 'never', 'more', 'less', 'above', 'below', 'down', 'up', 'further', 'again', 'very'}
        custom_stops = list((base_stops.union(fomc_chatter)) - preserve)

        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=3000, stop_words=custom_stops, min_df=2, sublinear_tf=True)
        self.df_train = None
        self.train_vectors = None

    def fit(self, df_train):
        self.df_train = df_train.copy().reset_index(drop=True)
        self.df_train['is_extreme'] = (self.df_train['pips_same_day'] >= self.pip_threshold).astype(int)
        self.train_vectors = self.vectorizer.fit_transform(self.df_train['speech_text'])

    def predict(self, speech_text, event_label="Event"):
        new_vec = self.vectorizer.transform([speech_text])
        sims = cosine_similarity(new_vec, self.train_vectors)[0]
        top_indices = np.argsort(sims)[::-1][:self.top_k]
        top_sims = sims[top_indices]
        top_labels = self.df_train.iloc[top_indices]['is_extreme'].values
        top_pips = self.df_train.iloc[top_indices]['pips_same_day'].values
        top_dates = self.df_train.iloc[top_indices]['date_str'].values

        weights = top_sims / (np.sum(top_sims) + 1e-9)
        prob = float(np.sum(weights * top_labels))
        expected_pips = float(np.sum(weights * top_pips))

        summary_table = pd.DataFrame({
            'Rank': range(1, self.top_k + 1),
            'Reference Meeting': top_dates,
            'Cosine Similarity': np.round(top_sims, 4),
            'Historical Realized Pips': top_pips,
            'Regime': ['Extreme' if x == 1 else 'Normal' for x in top_labels]
        })

        return {
            'probability': prob,
            'expected_pips': expected_pips,
            'summary_table': summary_table
        }


def run_pipeline():
    df_raw = load_all_fomc_transcripts('dataset')
    df_all = fetch_and_merge_gold_volatility(df_raw)

    print(f"Dataset Shape: {df_all.shape}")
    print(f"Sample Period: {df_all['date_str'].min()} to {df_all['date_str'].max()}")
    print("\nSummary Statistics (Volatility in Pips):")
    print(df_all[['pips_same_day', 'pips_extended']].describe().round(1))

    split_date = pd.to_datetime('2025-01-01')
    df_train = df_all[df_all['date'] < split_date].copy().reset_index(drop=True)
    df_test = df_all[df_all['date'] >= split_date].copy().reset_index(drop=True)

    print(f"\nTrain Split (2020-2024): N = {len(df_train)}")
    print(f"Test Split  (2025-2026): N = {len(df_test)}")

    THRESHOLD_PIPS = 400
    model = FOMCGoldVolatilityModel(pip_threshold=THRESHOLD_PIPS, top_k=5)
    model.fit(df_train)

    y_true = (df_test['pips_same_day'] >= THRESHOLD_PIPS).astype(int).values
    test_probs = [model.predict(row['speech_text'])['probability'] for _, row in df_test.iterrows()]
    test_probs = np.array(test_probs)

    DECISION_THRESHOLD = 0.53
    y_preds = (test_probs >= DECISION_THRESHOLD).astype(int)

    acc = accuracy_score(y_true, y_preds)
    roc_auc = roc_auc_score(y_true, test_probs)
    brier = brier_score_loss(y_true, test_probs)

    print("\nOUT-OF-SAMPLE STATISTICAL EVALUATION (2025-2026)")
    print("-" * 50)
    print(f"Binary Accuracy      : {acc * 100:.1f}%")
    print(f"Area Under ROC (AUC) : {roc_auc:.3f}")
    print(f"Brier Loss Score     : {brier:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_preds, target_names=['Normal Volatility', 'Extreme Volatility'], digits=3, zero_division=0))

    # Inference on latest meeting
    latest_event = df_test.iloc[-1]
    res = model.predict(latest_event['speech_text'], event_label=latest_event['date_str'])
    print(f"\nInference for Event {latest_event['date_str']}:")
    print(f"Probability of Extreme Volatility (>= {THRESHOLD_PIPS} pips): {res['probability'] * 100:.1f}%")
    print(f"Expected Volatility Range: {res['expected_pips']:.1f} pips (${res['expected_pips']/10:.2f})")
    print("\nNearest Historical Neighbors:")
    print(res['summary_table'].to_string(index=False))


if __name__ == '__main__':
    run_pipeline()
