# -*- coding: utf-8 -*-
# Pythonista / iOS — NumPy-only EEG -> H0 correlations
import os, csv, math, random
import numpy as np
import matplotlib.pyplot as plt

# === 1) CHEMIN RACINE (ajuste si besoin) =========================
BASE = "/private/var/mobile/Containers/Shared/AppGroup/4AD27D83-5289-46CD-B42C-CDE70593463D/Pythonista3/Documents/guitar.wav"
EEG_OUT = os.path.join(BASE, "EEG_ALAS_OUT")
H0_MAP = os.path.join(BASE, "H0_map.csv")
os.makedirs(EEG_OUT, exist_ok=True)

# === 2) OUTILS ====================================================
def read_alas_csv(path):
    """Lit un ALAS CSV : colonnes Time,MV1,MV2,MV3,MV4 (en-tête inclus).
       Retourne t (s) et signal mono (moyenne des MV1..MV4)."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV vide")
    # trouve l'entête (si la 1re cellule contient 'Time', on saute 1 ligne)
    start = 1 if rows[0] and rows[0][0].strip().lower().startswith("time") else 0
    t_list, y_list = [], []
    for r in rows[start:]:
        if not r or len(r) < 5:
            continue
        try:
            t = float(r[0])
            mv = [float(r[i]) for i in range(1, 5)]
        except ValueError:
            # ligne non-numérique -> ignore
            continue
        t_list.append(t)
        y_list.append(sum(mv) / 4.0)
    t = np.asarray(t_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    if t.size < 8:
        raise ValueError("Trop peu d'échantillons")
    # si temps non monotone (certains exports), on trie
    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)
        t, y = t[idx], y[idx]
    return t, y

def band_power_rel(y, fs, f_lo, f_hi):
    """Puissance relative dans [f_lo,f_hi] via fenêtre de Hann + FFT."""
    n = int(2**np.floor(np.log2(len(y))))  # puissance de 2
    y = y[:n]
    win = np.hanning(n)
    yf = np.fft.rfft(y * win)
    psd = (np.abs(yf) ** 2) / np.sum(win**2)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    total = np.trapz(psd, freqs)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if total <= 0 or not np.any(band):
        return 0.0
    return float(np.trapz(psd[band], freqs[band]) / total)

def dominant_freq(y, fs, fmin=1.0, fmax=60.0):
    n = int(2**np.floor(np.log2(len(y))))
    y = y[:n]
    win = np.hanning(n)
    yf = np.fft.rfft((y - np.mean(y)) * win)
    mag = np.abs(yf)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return np.nan
    i = np.argmax(mag[band])
    return float(freqs[band][i])

def snr_around(y, fs, f0=23.5, bw=0.5, fmin=1.0, fmax=60.0):
    """SNR (dB) = 10*log10(P_signal/P_noise) avec fenêtre étroite ±bw autour de f0."""
    n = int(2**np.floor(np.log2(len(y))))
    y = y[:n]
    win = np.hanning(n)
    yf = np.fft.rfft((y - np.mean(y)) * win)
    psd = (np.abs(yf) ** 2)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    main_band = (freqs >= max(fmin, f0 - bw)) & (freqs <= min(fmax, f0 + bw))
    side_band = (freqs >= fmin) & (freqs <= fmax) & (~main_band)
    P_sig = np.trapz(psd[main_band], freqs[main_band]) if np.any(main_band) else 0.0
    P_noise = np.trapz(psd[side_band], freqs[side_band]) if np.any(side_band) else 0.0
    if P_sig <= 0 or P_noise <= 0:
        return -150.0
    return float(10.0 * math.log10(P_sig / P_noise))

def guess_fs(t):
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise ValueError("Temps non monotone")
    return float(1.0 / dt)

def perm_test_corr(x, y, n_perm=5000, seed=123):
    rng = random.Random(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    r_obs = float(np.corrcoef(x, y)[0, 1])
    y_list = list(y)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(y_list)
        r_perm = float(np.corrcoef(x, np.array(y_list))[0, 1])
        if abs(r_perm) >= abs(r_obs):
            count += 1
    return (count + 1) / (n_perm + 1)

# === 3) SCAN + RÉSUMÉ ============================================
print("BASE =", BASE)
print("Exists:", os.path.isdir(BASE))
print("Scanning ALAS files...")

alas_files = sorted(
    [f for f in os.listdir(BASE) if f.startswith("ALAS_") and f.endswith(".csv")]
)
for f in alas_files:
    print("->", f)
if not alas_files:
    print("❌ Aucun fichier ALAS trouvé.")
    raise SystemExit(0)

summary_rows = [["file", "fdom_Hz", "SNR23_5_dB", "beta_rel", "gamma_low_rel"]]

for fn in alas_files:
    path = os.path.join(BASE, fn)
    try:
        t, y = read_alas_csv(path)
        fs = guess_fs(t)
        fdom = dominant_freq(y, fs, 1.0, 60.0)
        snr = snr_around(y, fs, 23.5, 0.5, 1.0, 60.0)
        beta = band_power_rel(y, fs, 12.0, 30.0)
        g_low = band_power_rel(y, fs, 30.0, 45.0)
        summary_rows.append([fn, f"{fdom:.2f}", f"{snr:.2f}", f"{beta:.3f}", f"{g_low:.3f}"])
    except Exception as e:
        print("⚠️", fn, "->", e)

os.makedirs(EEG_OUT, exist_ok=True)
summary_csv = os.path.join(EEG_OUT, "alas_summary.csv")
with open(summary_csv, "w", newline="") as fw:
    csv.writer(fw).writerows(summary_rows)
print("✅ Résumé EEG écrit ->", summary_csv)

# === 4) H0_map.csv : créer si absent =============================
if not os.path.isfile(H0_MAP):
    print("\nInfo: H0_map.csv absent. Je le crée pour toi, à remplir :")
    rows = [["file", "H0"]]
    for f in alas_files:
        rows.append([f, ""])
    with open(H0_MAP, "w", newline="") as fw:
        csv.writer(fw).writerows(rows)
    print("→", H0_MAP)
    try:
        import console, editor
        console.alert(
            "H0_map.csv créé",
            "Ouvre le fichier et remplis H0 (km/s/Mpc) pour chaque ligne, puis relance.",
            "Ouvrir"
        )
        editor.open_file(H0_MAP)
    except Exception:
        input("\nH0_map.csv créé. Appuie sur Entrée après lecture du message.")
    raise SystemExit(0)

# === 5) Join avec H0 et corrélations =============================
def load_csv_as_dict(path):
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]

sum_dicts = load_csv_as_dict(summary_csv)
h0_dicts  = load_csv_as_dict(H0_MAP)

# index par nom de fichier
h0_map = {}
for d in h0_dicts:
    f = (d.get("file") or "").strip()
    try:
        h0 = float(d.get("H0"))
    except Exception:
        continue
    if f:
        h0_map[f] = h0

joined = []
for d in sum_dicts:
    f = d.get("file")
    if not f:  # skip header row
        continue
    if f in h0_map:
        row = {
            "file": f,
            "H0": h0_map[f],
            "fdom_Hz": float(d["fdom_Hz"]),
            "SNR23_5_dB": float(d["SNR23_5_dB"]),
            "beta_rel": float(d["beta_rel"]),
            "gamma_low_rel": float(d["gamma_low_rel"]),
        }
        joined.append(row)

if not joined:
    print("i️ H0_map.csv trouvé mais sans valeurs utilisables. Remplis-le puis relance.")
    raise SystemExit(0)

# Corrélations
def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

H0 = [r["H0"] for r in joined]
F  = [r["fdom_Hz"] for r in joined]
S  = [r["SNR23_5_dB"] for r in joined]
B  = [r["beta_rel"] for r in joined]
G  = [r["gamma_low_rel"] for r in joined]

pairs = [
    ("fdom_Hz", F),
    ("SNR23_5_dB", S),
    ("beta_rel (12–30 Hz)", B),
    ("gamma_low_rel (30–45 Hz)", G),
]

# Sauve un CSV joint
joined_csv = os.path.join(EEG_OUT, "alas_h0_join.csv")
with open(joined_csv, "w", newline="") as fw:
    w = csv.writer(fw)
    w.writerow(["file","H0","fdom_Hz","SNR23_5_dB","beta_rel","gamma_low_rel"])
    for r in joined:
        w.writerow([r["file"], r["H0"], r["fdom_Hz"], r["SNR23_5_dB"], r["beta_rel"], r["gamma_low_rel"]])
print("✅ Joint EEG+H0 ->", joined_csv)

# Figures + stats
for name, X in pairs:
    r = corr(H0, X)
    p = perm_test_corr(H0, X, n_perm=5000, seed=123)
    print(f"corr(H0, {name}) = {r:.3f}  (perm p≈{p:.3f})")

    plt.figure()
    plt.scatter(X, H0)
    plt.xlabel(name)
    plt.ylabel("H0 (km/s/Mpc)")
    title = f"H0 vs {name} (corr={r:.3f}, n={len(H0)})"
    if not (p is np.nan):
        title += f"\nperm p≈{p:.3f}"
    plt.title(title)
    out_png = os.path.join(EEG_OUT, f"scatter_H0_vs_{name.split()[0].replace('/', '')}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print("→ PNG:", out_png)

print("🏁 Terminé.")
# ==== Diagnostics supplémentaires (sans SciPy) ====
import numpy as np, csv, os

JOIN = os.path.join(EEG_OUT, "alas_h0_join.csv")

def load_join(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "file": r["file"],
                    "H0": float(r["H0"]),
                    "fdom_Hz": float(r["fdom_Hz"]),
                    "SNR23_5_dB": float(r["SNR23_5_dB"]),
                    "beta_rel": float(r["beta_rel"]),
                    "gamma_low_rel": float(r.get("gamma_low_rel", r.get("gamma_rel", "nan")))
                })
            except Exception:
                pass
    return rows

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 2 or np.std(x)==0 or np.std(y)==0: return np.nan
    return float(np.corrcoef(x, y)[0,1])

def spearman(x, y):
    # corrélation de rangs (sans SciPy)
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 2: return np.nan
    def ranks(a):
        order = a.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        return r
    return pearson(ranks(x), ranks(y))

def loo_corr(x, y, corr_func):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    vals = []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        vals.append(corr_func(x[mask], y[mask]))
    return np.array(vals)

rows = load_join(JOIN)
if rows:
    metrics = ["fdom_Hz","SNR23_5_dB","beta_rel","gamma_low_rel"]
    H0 = np.array([r["H0"] for r in rows], float)
    print("\n=== DIAGNOSTICS ===")
    for m in metrics:
        X = np.array([r[m] for r in rows], float)
        r_p = pearson(H0, X)
        r_s = spearman(H0, X)
        loo_p = loo_corr(H0, X, pearson)
        loo_s = loo_corr(H0, X, spearman)
        print(f"{m:>12} | Pearson={r_p:+.3f}  Spearman={r_s:+.3f}  "
              f"LOO-P range [{np.nanmin(loo_p):+.3f},{np.nanmax(loo_p):+.3f}]  "
              f"LOO-S range [{np.nanmin(loo_s):+.3f},{np.nanmax(loo_s):+.3f}]")
