import os
import re
import sys
import pypdf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# Pastikan encoding console Windows rapi
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def extract_chair_speech_from_pdf(pdf_path):
    """
    Mengekstrak ucapan Ketua The Fed (baik Powell, Warsh, atau Yellen) dari file PDF.
    Mendeteksi pola: 'CHAIRMAN WARSH.', 'CHAIR POWELL.', 'CHAIR ...'
    """
    reader = pypdf.PdfReader(pdf_path)
    full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Deteksi nama Ketua dari judul atau awal teks
    detected_chair = "CHAIR"
    if "WARSH" in full_text[:1000].upper():
        detected_chair = "CHAIRMAN KEVIN WARSH"
    elif "POWELL" in full_text[:1000].upper():
        detected_chair = "CHAIR JEROME POWELL"

    # Pisahkan dialog berdasarkan nama pembicara yang diikuti titik
    # Contoh: CHAIRMAN WARSH., MICHELLE SMITH., NICK TIMIRAOS.
    lines = full_text.split('\n')
    chair_dialogues = []
    is_chair_speaking = False

    for line in lines:
        line_clean = line.strip()
        # Cek apakah baris ini adalah pergantian ke pembicara Ketua
        if re.match(r'^(CHAIR|CHAIRMAN)\s+[A-Z\s]+\.', line_clean):
            is_chair_speaking = True
            # Hapus label nama di awal
            text_part = re.sub(r'^(CHAIR|CHAIRMAN)\s+[A-Z\s]+\.\s*', '', line_clean)
            if text_part:
                chair_dialogues.append(text_part)
        elif re.match(r'^[A-Z\s]{3,35}\.', line_clean):
            # Pergantian ke jurnalis atau orang lain
            is_chair_speaking = False
        else:
            if is_chair_speaking and len(line_clean) > 0:
                chair_dialogues.append(line_clean)

    # Robust Fallback: Jika PDF berupa 3 halaman Opening Remarks murni tanpa tag nama per kalimat
    extracted_text = " ".join(chair_dialogues).strip()
    if len(extracted_text.split()) < 50:
        # Gunakan seluruh teks dokumen pembuka
        speech_final = full_text
    else:
        speech_final = extracted_text

    return detected_chair, speech_final


def main():
    print("=" * 70)
    print("      PENGUJIAN MODEL PADA TRANSKRIP PDF TERBARU (SEPTEMBER 2026)      ")
    print("=" * 70)

    # 1. Cari file PDF terbaru di folder 'transcript'
    transcript_folder = "transcript"
    if not os.path.exists(transcript_folder):
        print(f"[!] Folder '{transcript_folder}' tidak ditemukan.")
        return

    pdf_files = [f for f in os.listdir(transcript_folder) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"[!] Tidak ada file PDF di folder '{transcript_folder}'.")
        return

    # Urutkan berdasarkan waktu unduh / modifikasi file terbaru
    pdf_files.sort(key=lambda f: os.path.getmtime(os.path.join(transcript_folder, f)))
    latest_pdf = pdf_files[-1]
    latest_pdf_path = os.path.join(transcript_folder, latest_pdf)

    print(f"[*] Menemukan file PDF terbaru: {latest_pdf_path}")
    chair_name, speech_text = extract_chair_speech_from_pdf(latest_pdf_path)
    word_count = len(speech_text.split())

    print(f"[*] Ketua The Fed Terdeteksi : {chair_name}")
    print(f"[*] Jumlah Kata Ucapan Ketua : {word_count} kata diekstrak dari PDF.")

    # 2. Muat basis data historis dari dataset/
    print("\n[*] Menyelaraskan dengan 45 rapat The Fed masa lalu...")
    transcripts = []
    dataset_files = sorted([f for f in os.listdir('dataset') if f.endswith('.txt') and 'FOMC' in f])
    for f in dataset_files:
        m = re.search(r'(\d{8})', f)
        if not m: continue
        with open(os.path.join('dataset', f), 'r', encoding='utf-8', errors='ignore') as fp:
            raw = fp.read()
            dialogs = re.findall(r'<NAME>(.*?)</NAME>[\.\:\s]*(.*?)(?=<NAME>|$)', raw, re.DOTALL)
            p_text = " ".join([sp.strip() for name, sp in dialogs if 'POWELL' in name.upper() or 'WARSH' in name.upper()])
            if len(p_text) > 100:
                transcripts.append({'date_str': m.group(1), 'text': p_text})

    df_hist = pd.DataFrame(transcripts)

    # Muat data harga emas
    gold = pd.read_csv('gold_prices_cache.csv', index_col=0, parse_dates=True)
    pips_list = []
    for _, row in df_hist.iterrows():
        d = pd.to_datetime(row['date_str'], format='%Y%m%d')
        sub = gold.loc[gold.index >= d].head(1)
        if len(sub) > 0:
            pips = round(float(sub.iloc[0]['High'] - sub.iloc[0]['Low']) * 10, 1)
        else:
            pips = 350.0
        pips_list.append(pips)

    df_hist['pips'] = pips_list
    THRESHOLD = 400.0
    df_hist['is_extreme'] = (df_hist['pips'] >= THRESHOLD).astype(int)

    # 3. Ekstraksi Fitur TF-IDF N-Gram
    base_stops = set(ENGLISH_STOP_WORDS)
    fomc_chatter = {'chair', 'chairman', 'warsh', 'powell', 'thank', 'thanks', 'good', 'afternoon', 'morning', 'today', 'question', 'think'}
    preserve = {'not', 'no', 'never', 'more', 'less', 'above', 'below', 'down', 'up', 'further', 'again', 'very'}
    stops = list((base_stops.union(fomc_chatter)) - preserve)

    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=3000, stop_words=stops, min_df=2, sublinear_tf=True)
    hist_vecs = vec.fit_transform(df_hist['text'])

    # 4. Prediksi untuk Dokumen PDF Baru
    new_vec = vec.transform([speech_text])
    sims = cosine_similarity(new_vec, hist_vecs)[0]

    top_k = 5
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_sims = sims[top_idx]
    top_labels = df_hist.iloc[top_idx]['is_extreme'].values
    top_pips = df_hist.iloc[top_idx]['pips'].values
    top_dates = df_hist.iloc[top_idx]['date_str'].values

    weights = top_sims / (np.sum(top_sims) + 1e-9)
    prob = float(np.sum(weights * top_labels) * 100)
    perkiraan_pips = float(np.sum(weights * top_pips))
    perkiraan_dollar = perkiraan_pips / 10.0

    # 5. Tampilkan Hasil
    print("\n" + "=" * 70)
    print(f"   HASIL PREDIKSI VOLATILITAS: {latest_pdf} ({chair_name})   ")
    print("=" * 70)

    if prob >= 60:
        status = "⚠️  BAHAYA: POTENSI VOLATILITAS TINGGI (GERAKAN BESAR)"
        strategi = "Straddle Pending Order (Buy Stop & Sell Stop). Jangan open posisi instan!"
    elif prob >= 40:
        status = "⚡  WASPADA: VOLATILITAS SEDANG (GERAKAN NORMAL)"
        strategi = "Tunggu candle 15 menit selesai, target profit wajar."
    else:
        status = "🟢  AMAN: VOLATILITAS RENDAH (PASAR CENDERUNG TENANG)"
        strategi = "Pasar cenderung sideways, hindari mengejar lonjakan sesaat."

    print(f"Status Sinyal Model        : {status}")
    print(f"Probabilitas Gerak Ekstrem : {prob:.1f}% (Peluang pergerakan >= 400 pips)")
    print(f"Estimasi Rentang Harga Emas: ± {perkiraan_pips:.0f} Pips (± $ {perkiraan_dollar:.1f} USD)")
    print(f"Rekomendasi Eksekusi       : {strategi}")
    print("-" * 70)
    print("Top-5 Rapat Masa Lalu dengan Nada Bicara Paling Mirip:")
    for rank in range(top_k):
        tgl = pd.to_datetime(top_dates[rank], format='%Y%m%d').strftime('%Y-%m-%d')
        flag = "[EKSTREM]" if top_labels[rank] == 1 else "[NORMAL]"
        print(f"  {rank+1}. Tanggal: {tgl} | Cosine Sim: {top_sims[rank]:.4f} | Realisasi: {top_pips[rank]:>6.1f} pips | {flag}")
    print("=" * 70)

    # 6. Simpan hasil eksekusi ke file riwayat_prediksi.csv secara otomatis
    log_file = "riwayat_prediksi.csv"
    log_entry = {
        'Waktu_Eksekusi': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'File_PDF': latest_pdf,
        'Ketua_The_Fed': chair_name,
        'Probabilitas_Persen': round(prob, 1),
        'Estimasi_Pips': round(perkiraan_pips, 1),
        'Status_Sinyal': "BAHAYA" if prob >= 60 else ("WASPADA" if prob >= 40 else "AMAN")
    }
    df_log_entry = pd.DataFrame([log_entry])
    if os.path.exists(log_file):
        df_log_entry.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_log_entry.to_csv(log_file, mode='w', header=True, index=False)
    print(f"[*] Hasil eksekusi ini telah otomatis tersimpan ke berkas: {log_file}")


if __name__ == '__main__':
    main()
