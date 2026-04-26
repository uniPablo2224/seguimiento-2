"""
Monitor de RAM y CPU durante multiplicación de matrices.


Uso:
    python monitor_recursos.py            # corre con n=256 (demo rápido)
    python monitor_recursos.py --n 1024   # prueba más grande
    python monitor_recursos.py --n 512 --algo naiv winograd bloques
"""

import argparse
import os
import random
import threading
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import psutil

# ─────────────────────────────────────────────
# MONITOR EN HILO SEPARADO
# ─────────────────────────────────────────────

class ResourceMonitor:
    """Muestrea CPU y RAM cada `interval` segundos en un hilo daemon."""

    def __init__(self, interval=0.2):
        self.interval  = interval
        self.timestamps = []   # segundos desde inicio
        self.cpu        = []   # porcentaje CPU (promedio todos los núcleos)
        self.ram_used   = []   # GB usados
        self.ram_total  = psutil.virtual_memory().total / 1e9
        self._running   = False
        self._thread    = None
        self._start_t   = None
        self.events     = []   # (tiempo_s, etiqueta) para marcar inicio/fin de algos

    def start(self):
        self._running = True
        self._start_t = time.perf_counter()
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        # primera llamada a cpu_percent descartada (calibración)
        psutil.cpu_percent(interval=None)
        while self._running:
            t   = time.perf_counter() - self._start_t
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().used / 1e9
            self.timestamps.append(t)
            self.cpu.append(cpu)
            self.ram_used.append(ram)
            time.sleep(self.interval)

    def mark(self, label):
        """Registra un evento con timestamp actual."""
        t = time.perf_counter() - self._start_t
        self.events.append((t, label))

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self):
        if not self.cpu:
            return {}
        return {
            "cpu_max":    max(self.cpu),
            "cpu_mean":   sum(self.cpu) / len(self.cpu),
            "ram_max_gb": max(self.ram_used),
            "ram_total_gb": self.ram_total,
            "ram_max_pct": max(self.ram_used) / self.ram_total * 100,
            "duracion_s": self.timestamps[-1] if self.timestamps else 0,
        }


# ─────────────────────────────────────────────
# ALGORITMOS DE PRUEBA
# ─────────────────────────────────────────────

def gen_matrix_np(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(100_000, 1_000_000, size=(n, n), dtype=np.int64)


def naiv_python(A_list, B_list):
    """Naive puro en Python — O(n³). Solo para n pequeño."""
    n = len(A_list)
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            aik = A_list[i][k]
            for j in range(n):
                C[i][j] += aik * B_list[k][j]
    return C


def winograd_np(A, B):
    """Winograd vectorizado con NumPy."""
    n = A.shape[0]
    half = n // 2
    row_f = np.sum(A[:, 0:2*half:2] * A[:, 1:2*half:2], axis=1)
    col_f = np.sum(B[0:2*half:2, :] * B[1:2*half:2, :], axis=0)
    C = -row_f[:, None] - col_f[None, :]
    for k in range(half):
        C += np.outer(A[:, 2*k] + B[2*k+1, :], A[:, 2*k+1] + B[2*k, :])
    if n % 2:
        C += np.outer(A[:, -1], B[-1, :])
    return C


def bloques_np(A, B, bs=256):
    """Multiplicación por bloques con NumPy (@)."""
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.int64)
    for i in range(0, n, bs):
        for k in range(0, n, bs):
            for j in range(0, n, bs):
                C[i:i+bs, j:j+bs] += A[i:i+bs, k:k+bs] @ B[k:k+bs, j:j+bs]
    return C


def numpy_dot(A, B):
    """np.dot — referencia BLAS."""
    return A @ B


# ─────────────────────────────────────────────
# EJECUCIÓN CON MONITOREO
# ─────────────────────────────────────────────

SUITE = {
    "naiv_python":  ("Naive Python puro",       None),
    "winograd_np":  ("Winograd NumPy",           None),
    "bloques_np":   ("Bloques NumPy (bs=256)",   None),
    "numpy_dot":    ("NumPy @ (BLAS)",           None),
}

def run_suite(n, algos_sel, monitor):
    A_np = gen_matrix_np(n, seed=1)
    B_np = gen_matrix_np(n, seed=2)

    # Convertir a lista solo si se necesita naive
    A_list = A_np.tolist() if "naiv_python" in algos_sel else None
    B_list = B_np.tolist() if "naiv_python" in algos_sel else None

    results = []
    fns = {
        "naiv_python": lambda: naiv_python(A_list, B_list),
        "winograd_np": lambda: winograd_np(A_np, B_np),
        "bloques_np":  lambda: bloques_np(A_np, B_np),
        "numpy_dot":   lambda: numpy_dot(A_np, B_np),
    }

    for key in algos_sel:
        label, _ = SUITE[key]
        print(f"  → {label:<30}", end="", flush=True)
        monitor.mark(f"▶ {label}")
        t0 = time.perf_counter()
        fns[key]()
        elapsed = time.perf_counter() - t0
        monitor.mark(f"■ {label}")
        print(f"  {elapsed:.4f} s")
        results.append({"key": key, "label": label, "tiempo_s": elapsed})

    return results


# ─────────────────────────────────────────────
# GRÁFICAS
# ─────────────────────────────────────────────

COLORS = {
    "naiv_python": "#E24B4A",
    "winograd_np": "#185FA5",
    "bloques_np":  "#2E9E6B",
    "numpy_dot":   "#9B59B6",
}

def plot_monitor(monitor, results, n, output_dir="resultados_monitor"):
    os.makedirs(output_dir, exist_ok=True)

    ts  = monitor.timestamps
    cpu = monitor.cpu
    ram = monitor.ram_used
    s   = monitor.summary()

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Colores por evento (inicio/fin de cada algo)
    event_colors = {}
    idx = 0
    color_list = ["#E24B4A","#185FA5","#2E9E6B","#9B59B6"]
    for t, label in monitor.events:
        key = label[2:]   # quitar "▶ " o "■ "
        if key not in event_colors:
            event_colors[key] = color_list[idx % len(color_list)]
            idx += 1

    def shade_events(ax):
        """Sombrea el período de cada algoritmo."""
        starts = {lbl[2:]: t for t, lbl in monitor.events if lbl.startswith("▶")}
        ends   = {lbl[2:]: t for t, lbl in monitor.events if lbl.startswith("■")}
        for key, t0 in starts.items():
            t1 = ends.get(key, ts[-1])
            ax.axvspan(t0, t1, alpha=0.10, color=event_colors.get(key, "#888888"), zorder=0)
            ax.axvline(t0, color=event_colors.get(key, "#888888"), linewidth=1, linestyle="--", alpha=0.7)
            ax.text(t0, ax.get_ylim()[1]*0.97, key.split("(")[0].strip(),
                    fontsize=6.5, rotation=90, va="top", ha="left",
                    color=event_colors.get(key, "#555555"))

    # ── 1. CPU en el tiempo ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#F8F9FA")
    ax1.plot(ts, cpu, color="#185FA5", linewidth=1.2, zorder=3)
    ax1.fill_between(ts, cpu, alpha=0.15, color="#185FA5", zorder=2)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("CPU (%)", fontsize=10)
    ax1.set_title(f"Uso de CPU durante la ejecución — n={n}", fontsize=11, fontweight="bold")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    shade_events(ax1)
    ax1.set_xlabel("Tiempo (s)", fontsize=9)

    # ── 2. RAM en el tiempo ──
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor("#F8F9FA")
    ax2.plot(ts, ram, color="#E24B4A", linewidth=1.2, zorder=3)
    ax2.fill_between(ts, ram, alpha=0.15, color="#E24B4A", zorder=2)
    ax2.axhline(monitor.ram_total, color="#888888", linewidth=0.8,
                linestyle=":", label=f"RAM total ({monitor.ram_total:.1f} GB)")
    ax2.set_ylabel("RAM usada (GB)", fontsize=10)
    ax2.set_title(f"Uso de RAM durante la ejecución — n={n}", fontsize=11, fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="upper right")
    shade_events(ax2)
    ax2.set_xlabel("Tiempo (s)", fontsize=9)

    # ── 3. Tiempos de ejecución (barras) ──
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor("#F8F9FA")
    names  = [r["label"] for r in results]
    times  = [r["tiempo_s"] for r in results]
    keys   = [r["key"] for r in results]
    cols   = [COLORS.get(k, "#888888") for k in keys]
    bars   = ax3.bar(range(len(names)), times, color=cols, alpha=0.85)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([n.split("(")[0].strip() for n in names], rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Tiempo (s)", fontsize=10)
    ax3.set_title("Tiempo total por algoritmo", fontsize=11, fontweight="bold")
    ax3.set_yscale("log")
    for bar, val in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.2,
                 f"{val:.4f}s", ha="center", va="bottom", fontsize=8)
    ax3.yaxis.grid(True, which="both", linestyle="--", alpha=0.4)

    # ── 4. Resumen estadístico ──
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor("#F8F9FA")
    ax4.axis("off")
    resumen = [
        ["Métrica", "Valor"],
        ["n (tamaño de matriz)", str(n)],
        ["Memoria por matriz (int64)", f"{n*n*8/1e6:.1f} MB"],
        ["3 matrices en RAM", f"{3*n*n*8/1e6:.1f} MB"],
        ["RAM total del sistema", f"{s.get('ram_total_gb',0):.1f} GB"],
        ["RAM máx. usada", f"{s.get('ram_max_gb',0):.2f} GB  ({s.get('ram_max_pct',0):.1f}%)"],
        ["CPU máx.", f"{s.get('cpu_max',0):.1f}%"],
        ["CPU promedio", f"{s.get('cpu_mean',0):.1f}%"],
        ["Duración total", f"{s.get('duracion_s',0):.2f} s"],
    ]
    table = ax4.table(cellText=resumen[1:], colLabels=resumen[0],
                      loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1F497D")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#DEEAF1")
        cell.set_edgecolor("#CCCCCC")
    ax4.set_title("Resumen de recursos", fontsize=11, fontweight="bold", pad=10)

    fig.suptitle(
        f"Monitor de Recursos — Multiplicación de Matrices n={n}\n"
        f"Universidad del Quindío — Seguimiento 2",
        fontsize=13, fontweight="bold", y=0.98
    )

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    png = os.path.join(output_dir, f"monitor_n{n}_{ts_str}.png")
    pdf = os.path.join(output_dir, f"monitor_n{n}_{ts_str}.pdf")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Gráficas guardadas:")
    print(f"    {png}")
    print(f"    {pdf}")
    return png, pdf


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Monitor de CPU/RAM durante multiplicación de matrices")
    p.add_argument("--n",    type=int, default=512,
                   help="Tamaño de la matriz n×n (default: 256)")
    p.add_argument("--algo", nargs="+",
                   choices=list(SUITE.keys()), default=list(SUITE.keys()),
                   help="Algoritmos a ejecutar (default: todos)")
    p.add_argument("--interval", type=float, default=0.2,
                   help="Intervalo de muestreo en segundos (default: 0.2)")
    p.add_argument("--out", type=str, default="resultados_monitor",
                   help="Directorio de salida")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n    = args.n

    # Advertir si naive es muy grande
    if "naiv_python" in args.algo and n > 512:
        print(f"  ⚠  Naive Python con n={n} puede tardar mucho. "
              "Considera omitirlo con --algo winograd_np bloques_np numpy_dot")

    print(f"\n{'='*60}")
    print(f"  Monitor de Recursos — n={n}")
    print(f"  Algoritmos: {', '.join(args.algo)}")
    print(f"  RAM total: {psutil.virtual_memory().total/1e9:.1f} GB")
    print(f"  Núcleos CPU: {psutil.cpu_count()}")
    print(f"{'='*60}\n")

    monitor = ResourceMonitor(interval=args.interval)
    monitor.start()

    try:
        results = run_suite(n, args.algo, monitor)
    finally:
        monitor.stop()

    s = monitor.summary()
    print(f"\n{'─'*60}")
    print(f"  CPU máx: {s['cpu_max']:.1f}%    CPU media: {s['cpu_mean']:.1f}%")
    print(f"  RAM máx: {s['ram_max_gb']:.2f} GB  ({s['ram_max_pct']:.1f}% de {s['ram_total_gb']:.1f} GB)")
    print(f"  Duración total: {s['duracion_s']:.2f} s")
    print(f"{'─'*60}")

    plot_monitor(monitor, results, n, output_dir=args.out)
    print("\n¡Listo!\n")