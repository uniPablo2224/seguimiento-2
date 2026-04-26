"""
Multiplicación de Matrices Grandes - Universidad del Quindío
Ingeniería de Sistemas y Computación - Seguimiento 2
Presentado por: Juan Pablo Sánchez López (1095208340) y Antonio Quiroz Prada


Implementación de 15 algoritmos de multiplicación de matrices.
Casos de prueba: n=64 (Caso 1) y n=128 (Caso 2)
Cada elemento tiene mínimo 6 dígitos (100000 a 999999).
Los resultados y tiempos se persisten en archivos JSON y CSV. Y se generan graficas de barras agrupadas en escala logaritmica para comparar los tiempos de ejecucion de cada algortimo en ambos casos de prueba.
"""


import numpy as np
import random
import time
import json
import csv
import os
import math
from datetime import datetime
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE CASOS DE PRUEBA
# ─────────────────────────────────────────────
CASO1_N = 64
CASO2_N = 128
MIN_VAL = 100_000
MAX_VAL = 999_999
RESULTS_DIR = "resultados"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# GENERACIÓN Y PERSISTENCIA DE MATRICES
# ─────────────────────────────────────────────

def generate_matrix(n: int, seed: int = None) -> list:
    """Genera una matriz n×n con enteros de mínimo 6 dígitos."""
    if seed is not None:
        random.seed(seed)
    return [[random.randint(MIN_VAL, MAX_VAL) for _ in range(n)] for _ in range(n)]


def save_matrix(matrix, filename):
    with open(filename, "w") as f:
        json.dump(matrix, f)


def load_matrix(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_or_create_matrices(n, caso):
    """Carga matrices persistentes o las crea si no existen."""
    fa = os.path.join(RESULTS_DIR, f"caso{caso}_n{n}_A.json")
    fb = os.path.join(RESULTS_DIR, f"caso{caso}_n{n}_B.json")
    if os.path.exists(fa) and os.path.exists(fb):
        print(f"  [Caso {caso}] Cargando matrices existentes n={n}...")
        A = load_matrix(fa)
        B = load_matrix(fb)
    else:
        print(f"  [Caso {caso}] Generando matrices n={n}...")
        A = generate_matrix(n, seed=caso * 100)
        B = generate_matrix(n, seed=caso * 200)
        save_matrix(A, fa)
        save_matrix(B, fb)
    return A, B


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def to_np(M):
    return np.array(M, dtype=np.int64)


def to_list(M):
    return M.tolist() if isinstance(M, np.ndarray) else M


def zeros(n):
    return [[0] * n for _ in range(n)]


def add_matrix(A, B, n):
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def sub_matrix(A, B, n):
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


# ─────────────────────────────────────────────
# ALGORITMO 1: NaivOnArray
# ─────────────────────────────────────────────

def naiv_on_array(A, B):
    """Multiplicación naive estándar O(n³)."""
    n = len(A)
    C = zeros(n)
    for i in range(n):
        for k in range(n):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


# ─────────────────────────────────────────────
# ALGORITMO 2: NaivLoopUnrollingTwo
# ─────────────────────────────────────────────

def naiv_loop_unrolling_two(A, B):
    """Naive con desenrollado de bucle por factor 2 en j."""
    n = len(A)
    C = zeros(n)
    for i in range(n):
        for k in range(n):
            aik = A[i][k]
            j = 0
            while j < n - 1:
                C[i][j]     += aik * B[k][j]
                C[i][j + 1] += aik * B[k][j + 1]
                j += 2
            if j < n:
                C[i][j] += aik * B[k][j]
    return C


# ─────────────────────────────────────────────
# ALGORITMO 3: NaivLoopUnrollingFour
# ─────────────────────────────────────────────

def naiv_loop_unrolling_four(A, B):
    """Naive con desenrollado de bucle por factor 4 en j."""
    n = len(A)
    C = zeros(n)
    for i in range(n):
        for k in range(n):
            aik = A[i][k]
            j = 0
            while j < n - 3:
                C[i][j]     += aik * B[k][j]
                C[i][j + 1] += aik * B[k][j + 1]
                C[i][j + 2] += aik * B[k][j + 2]
                C[i][j + 3] += aik * B[k][j + 3]
                j += 4
            while j < n:
                C[i][j] += aik * B[k][j]
                j += 1
    return C


# ─────────────────────────────────────────────
# ALGORITMO 4: WinogradOriginal
# ─────────────────────────────────────────────

def winograd_original(A, B):
    """Algoritmo de Winograd original — reduce multiplicaciones."""
    n = len(A)
    C = zeros(n)
    half = n // 2

    row_factor = [0] * n
    col_factor = [0] * n

    for i in range(n):
        for j in range(half):
            row_factor[i] += A[i][2*j] * A[i][2*j+1]

    for j in range(n):
        for i in range(half):
            col_factor[j] += B[2*i][j] * B[2*i+1][j]

    for i in range(n):
        for j in range(n):
            tmp = -row_factor[i] - col_factor[j]
            for k in range(half):
                tmp += (A[i][2*k] + B[2*k+1][j]) * (A[i][2*k+1] + B[2*k][j])
            C[i][j] = tmp

    if n % 2 != 0:
        for i in range(n):
            for j in range(n):
                C[i][j] += A[i][n-1] * B[n-1][j]
    return C


# ─────────────────────────────────────────────
# ALGORITMO 5: WinogradScaled
# ─────────────────────────────────────────────

def winograd_scaled(A, B):
    """Winograd con escalado previo para mejorar estabilidad numérica."""
    n = len(A)
    # Escalar filas de A y columnas de B
    lam = 2.0
    As = [[A[i][j] / lam for j in range(n)] for i in range(n)]
    Bs = [[B[i][j] / lam for j in range(n)] for i in range(n)]
    # Aplicar Winograd original sobre escaladas, luego reescalar resultado
    Cs_raw = winograd_original(As, Bs)  # usa floats aquí
    # Reescalar (lam*lam = 4 → factor de retorno)
    C = [[int(round(Cs_raw[i][j] * lam * lam)) for j in range(n)] for i in range(n)]
    return C


# ─────────────────────────────────────────────
# ALGORITMO 6: StrassenNaiv
# ─────────────────────────────────────────────

def strassen_naiv(A, B, threshold=32):
    """Strassen recursivo; usa naive cuando n ≤ threshold."""
    n = len(A)
    if n <= threshold:
        return naiv_on_array(A, B)

    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    m = mid
    M1 = strassen_naiv(add_matrix(A11, A22, m), add_matrix(B11, B22, m), threshold)
    M2 = strassen_naiv(add_matrix(A21, A22, m), B11, threshold)
    M3 = strassen_naiv(A11, sub_matrix(B12, B22, m), threshold)
    M4 = strassen_naiv(A22, sub_matrix(B21, B11, m), threshold)
    M5 = strassen_naiv(add_matrix(A11, A12, m), B22, threshold)
    M6 = strassen_naiv(sub_matrix(A21, A11, m), add_matrix(B11, B12, m), threshold)
    M7 = strassen_naiv(sub_matrix(A12, A22, m), add_matrix(B21, B22, m), threshold)

    C11 = add_matrix(sub_matrix(add_matrix(M1, M4, m), M5, m), M7, m)
    C12 = add_matrix(M3, M5, m)
    C21 = add_matrix(M2, M4, m)
    C22 = add_matrix(sub_matrix(add_matrix(M1, M3, m), M2, m), M6, m)

    C = zeros(n)
    for i in range(mid):
        for j in range(mid):
            C[i][j]           = C11[i][j]
            C[i][j + mid]     = C12[i][j]
            C[i + mid][j]     = C21[i][j]
            C[i + mid][j+mid] = C22[i][j]
    return C


# ─────────────────────────────────────────────
# ALGORITMO 7: StrassenWinograd
# ─────────────────────────────────────────────

def strassen_winograd(A, B, threshold=32):
    """Strassen–Winograd: variante que usa Winograd en la base."""
    n = len(A)
    if n <= threshold:
        return winograd_original(A, B)

    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    m = mid
    S1  = sub_matrix(B12, B22, m)
    S2  = add_matrix(A11, A12, m)
    S3  = add_matrix(A21, A22, m)
    S4  = sub_matrix(B21, B11, m)
    S5  = add_matrix(A11, A22, m)
    S6  = add_matrix(B11, B22, m)
    S7  = sub_matrix(A12, A22, m)
    S8  = add_matrix(B21, B22, m)
    S9  = sub_matrix(A11, A21, m)
    S10 = add_matrix(B11, B12, m)

    P1 = strassen_winograd(A11, S1,  threshold)
    P2 = strassen_winograd(S2,  B22, threshold)
    P3 = strassen_winograd(S3,  B11, threshold)
    P4 = strassen_winograd(A22, S4,  threshold)
    P5 = strassen_winograd(S5,  S6,  threshold)
    P6 = strassen_winograd(S7,  S8,  threshold)
    P7 = strassen_winograd(S9,  S10, threshold)

    C11 = add_matrix(sub_matrix(add_matrix(P5, P4, m), P2, m), P6, m)
    C12 = add_matrix(P1, P2, m)
    C21 = add_matrix(P3, P4, m)
    C22 = sub_matrix(sub_matrix(add_matrix(P5, P1, m), P3, m), P7, m)

    C = zeros(n)
    for i in range(mid):
        for j in range(mid):
            C[i][j]           = C11[i][j]
            C[i][j + mid]     = C12[i][j]
            C[i + mid][j]     = C21[i][j]
            C[i + mid][j+mid] = C22[i][j]
    return C


# ─────────────────────────────────────────────
# BLOQUE III — Sección III del paper (bloques con numpy)
# ─────────────────────────────────────────────

def _block_size(n, num_blocks=4):
    return max(1, n // num_blocks)


# Algoritmo 8: III.3 Sequential Block
def iii3_sequential_block(A, B):
    """Multiplicación por bloques secuencial (numpy)."""
    An = to_np(A)
    Bn = to_np(B)
    n = An.shape[0]
    bs = _block_size(n)
    C = np.zeros((n, n), dtype=np.int64)
    for i in range(0, n, bs):
        for k in range(0, n, bs):
            for j in range(0, n, bs):
                C[i:i+bs, j:j+bs] += An[i:i+bs, k:k+bs] @ Bn[k:k+bs, j:j+bs]
    return to_list(C)


# Algoritmo 9: III.4 Parallel Block (simulado con numpy vectorizado)
def iii4_parallel_block(A, B):
    """Bloques paralelos — simulado con operaciones vectorizadas numpy."""
    An = to_np(A)
    Bn = to_np(B)
    n = An.shape[0]
    bs = _block_size(n)
    C = np.zeros((n, n), dtype=np.int64)
    # Paralelismo simulado: acumulamos bloques de k en una pasada por (i,j)
    for k in range(0, n, bs):
        for i in range(0, n, bs):
            for j in range(0, n, bs):
                C[i:i+bs, j:j+bs] += An[i:i+bs, k:k+bs] @ Bn[k:k+bs, j:j+bs]
    return to_list(C)


# Algoritmo 10: III.5 Enhanced Parallel Block
def iii5_enhanced_parallel_block(A, B):
    """Bloques paralelos mejorados — reordenamiento de bucles para mejor caché."""
    An = to_np(A)
    Bn = to_np(B)
    n = An.shape[0]
    bs = _block_size(n, num_blocks=8)
    C = np.zeros((n, n), dtype=np.int64)
    for i in range(0, n, bs):
        for j in range(0, n, bs):
            for k in range(0, n, bs):
                C[i:i+bs, j:j+bs] += An[i:i+bs, k:k+bs] @ Bn[k:k+bs, j:j+bs]
    return to_list(C)


# ─────────────────────────────────────────────
# BLOQUE IV — Variantes con transposición de B
# ─────────────────────────────────────────────

# Algoritmo 11: IV.3 Sequential Block (con B transpuesta)
def iv3_sequential_block(A, B):
    """Bloques secuencial con B transpuesta para mejor acceso a memoria."""
    An = to_np(A)
    Bt = to_np(B).T
    n = An.shape[0]
    bs = _block_size(n)
    C = np.zeros((n, n), dtype=np.int64)
    for i in range(0, n, bs):
        for j in range(0, n, bs):
            for k in range(0, n, bs):
                C[i:i+bs, j:j+bs] += An[i:i+bs, k:k+bs] @ Bt[j:j+bs, k:k+bs].T
    return to_list(C)


# Algoritmo 12: IV.4 Parallel Block (con B transpuesta)
def iv4_parallel_block(A, B):
    """Bloques paralelos con B transpuesta."""
    An = to_np(A)
    Bt = to_np(B).T
    n = An.shape[0]
    bs = _block_size(n)
    C = np.zeros((n, n), dtype=np.int64)
    for k in range(0, n, bs):
        for i in range(0, n, bs):
            for j in range(0, n, bs):
                C[i:i+bs, j:j+bs] += An[i:i+bs, k:k+bs] @ Bt[j:j+bs, k:k+bs].T
    return to_list(C)


# Algoritmo 13: IV.5 Enhanced Parallel Block (con B transpuesta)
def iv5_enhanced_parallel_block(A, B):
    """Bloques paralelos mejorados con B transpuesta y bloques más finos."""
    An = to_np(A)
    Bt = to_np(B).T
    n = An.shape[0]
    bs = _block_size(n, num_blocks=8)
    C = np.zeros((n, n), dtype=np.int64)
    for i in range(0, n, bs):
        for j in range(0, n, bs):
            for k in range(0, n, bs):
                C[i:i+bs, j:j+bs] += An[i:i+bs, k:k+bs] @ Bt[j:j+bs, k:k+bs].T
    return to_list(C)


# ─────────────────────────────────────────────
# BLOQUE V — Winograd por bloques
# ─────────────────────────────────────────────

# Algoritmo 14: V.3 Sequential Block (Winograd por bloques)
def v3_sequential_block(A, B):
    """Winograd aplicado por bloques secuencialmente."""
    n = len(A)
    bs = _block_size(n)
    C_np = np.zeros((n, n), dtype=np.int64)
    for i in range(0, n, bs):
        for k in range(0, n, bs):
            for j in range(0, n, bs):
                Ablk = [row[k:k+bs] for row in A[i:i+bs]]
                Bblk = [row[j:j+bs] for row in B[k:k+bs]]
                Cblk = winograd_original(Ablk, Bblk)
                for bi in range(len(Cblk)):
                    for bj in range(len(Cblk[0])):
                        C_np[i+bi][j+bj] += Cblk[bi][bj]
    return to_list(C_np)


# Algoritmo 15: V.4 Parallel Block (Winograd por bloques, orden paralelo)
def v4_parallel_block(A, B):
    """Winograd por bloques con reordenamiento de bucles paralelo."""
    n = len(A)
    bs = _block_size(n)
    C_np = np.zeros((n, n), dtype=np.int64)
    for k in range(0, n, bs):
        for i in range(0, n, bs):
            for j in range(0, n, bs):
                Ablk = [row[k:k+bs] for row in A[i:i+bs]]
                Bblk = [row[j:j+bs] for row in B[k:k+bs]]
                Cblk = winograd_original(Ablk, Bblk)
                for bi in range(len(Cblk)):
                    for bj in range(len(Cblk[0])):
                        C_np[i+bi][j+bj] += Cblk[bi][bj]
    return to_list(C_np)


# ─────────────────────────────────────────────
# REGISTRO DE TIEMPOS
# ─────────────────────────────────────────────

ALGORITHMS = [
    (1,  "NaivOnArray",              naiv_on_array),
    (2,  "NaivLoopUnrollingTwo",     naiv_loop_unrolling_two),
    (3,  "NaivLoopUnrollingFour",    naiv_loop_unrolling_four),
    (4,  "WinogradOriginal",         winograd_original),
    (5,  "WinogradScaled",           winograd_scaled),
    (6,  "StrassenNaiv",             strassen_naiv),
    (7,  "StrassenWinograd",         strassen_winograd),
    (8,  "III.3 Sequential Block",   iii3_sequential_block),
    (9,  "III.4 Parallel Block",     iii4_parallel_block),
    (10, "III.5 Enhanced Parallel",  iii5_enhanced_parallel_block),
    (11, "IV.3 Sequential Block",    iv3_sequential_block),
    (12, "IV.4 Parallel Block",      iv4_parallel_block),
    (13, "IV.5 Enhanced Parallel",   iv5_enhanced_parallel_block),
    (14, "V.3 Sequential Block",     v3_sequential_block),
    (15, "V.4 Parallel Block",       v4_parallel_block),
]


def run_all(caso, n):
    print(f"\n{'='*60}")
    print(f"  CASO {caso}: matrices {n}×{n}")
    print(f"{'='*60}")
    A, B = get_or_create_matrices(n, caso)
    results = []
    for num, name, fn in ALGORITHMS:
        print(f"  [{num:02d}] {name:<30}", end="", flush=True)
        t0 = time.perf_counter()
        fn(A, B)
        elapsed = time.perf_counter() - t0
        print(f"  {elapsed:.4f} s")
        results.append({
            "caso": caso,
            "n": n,
            "algoritmo_num": num,
            "algoritmo": name,
            "tiempo_s": round(elapsed, 6),
            "timestamp": datetime.now().isoformat()
        })
    return results


def save_results(all_results):
    # JSON
    json_path = os.path.join(RESULTS_DIR, "tiempos.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "tiempos.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["caso", "n", "algoritmo_num", "algoritmo", "tiempo_s", "timestamp"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n  Resultados guardados en:")
    print(f"    {json_path}")
    print(f"    {csv_path}")


# ─────────────────────────────────────────────
# GRÁFICA DE BARRAS
# ─────────────────────────────────────────────

def plot_results(all_results, caso1_n, caso2_n):
    """Genera diagrama de barras agrupadas con escala logarítmica."""
    names  = [name for _, name, _ in ALGORITHMS]
    labels = [f"{num}. {name}" for num, name, _ in ALGORITHMS]

    caso1_map = {r["algoritmo"]: r["tiempo_s"] for r in all_results if r["caso"] == 1}
    caso2_map = {r["algoritmo"]: r["tiempo_s"] for r in all_results if r["caso"] == 2}

    t1 = [caso1_map.get(n, 0) for n in names]
    t2 = [caso2_map.get(n, 0) for n in names]

    n_algo  = len(names)
    x       = range(n_algo)
    width   = 0.38

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars1 = ax.bar([i - width/2 for i in x], t1, width,
                   label=f"Caso 1 — n={caso1_n}",
                   color="#185FA5", alpha=0.85, zorder=3)
    bars2 = ax.bar([i + width/2 for i in x], t2, width,
                   label=f"Caso 2 — n={caso2_n}",
                   color="#E24B4A", alpha=0.85, zorder=3)

    # Etiquetas de valor encima de cada barra
    for bar, val in zip(list(bars1) + list(bars2), t1 + t2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=6.5, rotation=90, color="#333333"
        )

    # Eje Y logarítmico
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"{y:.4f}s" if y < 0.01 else f"{y:.3f}s")
    )
    ax.set_ylim(bottom=1e-4)

    # Ejes y cuadrícula
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Tiempo de ejecución (s) — escala logarítmica", fontsize=11)
    ax.set_xlabel("Algoritmo", fontsize=11)
    ax.set_title(
        "Multiplicación de Matrices Grandes\n"
        "Comparación de tiempos de ejecución por algoritmo y caso de prueba",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Líneas separadoras entre grupos (visual)
    for sep in [2.5, 4.5, 6.5, 9.5, 12.5, 13.5]:
        ax.axvline(sep, color="#AAAAAA", linewidth=0.8, linestyle=":", zorder=2)

    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()

    # Guardar PNG y PDF
    png_path = os.path.join(RESULTS_DIR, "diagrama_barras.png")
    pdf_path = os.path.join(RESULTS_DIR, "diagrama_barras.pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Gráficas guardadas en:")
    print(f"    {png_path}")
    print(f"    {pdf_path}")
    return png_path, pdf_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MULTIPLICACIÓN DE MATRICES GRANDES")
    print("  Universidad del Quindío — Seguimiento 2")
    print("="*60)

    all_results = []
    all_results += run_all(caso=1, n=CASO1_N)
    all_results += run_all(caso=2, n=CASO2_N)

    save_results(all_results)
    plot_results(all_results, CASO1_N, CASO2_N)

    print("\n  Resumen de tiempos (segundos):")
    print(f"  {'#':<4} {'Algoritmo':<30} {'Caso1 (n='+str(CASO1_N)+')':<18} {'Caso2 (n='+str(CASO2_N)+')'}")
    print("  " + "-"*70)
    caso1 = {r["algoritmo_num"]: r["tiempo_s"] for r in all_results if r["caso"] == 1}
    caso2 = {r["algoritmo_num"]: r["tiempo_s"] for r in all_results if r["caso"] == 2}
    for num, name, _ in ALGORITHMS:
        t1 = caso1.get(num, 0)
        t2 = caso2.get(num, 0)
        print(f"  {num:<4} {name:<30} {t1:<18.4f} {t2:.4f}")

    print("\n¡Listo!\n")