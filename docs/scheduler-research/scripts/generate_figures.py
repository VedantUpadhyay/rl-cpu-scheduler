"""Generate 4 publication-quality figures for the RL-CPU-Scheduler paper."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURES_DIR = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone/docs/scheduler-research/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Pareto Frontier (W14-ω omega sweep)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_pareto_frontier():
    omega_s    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mct        = [20.51, 20.57, 21.03, 20.75, 20.71, 20.73, 20.49, 20.60, 20.43, 20.45, 20.36]
    starvation = [44.0, 42.0, 43.0, 39.5, 39.5, 38.0, 35.0, 32.5, 34.5, 38.5, 38.5]

    mlfq_mct   = 21.59;  mlfq_starv = 36.0
    w10c_mct   = 17.23;  w10c_starv = 0.0
    w12_mct    = 21.67;  w12_starv  = 53.2

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Shade region that dominates MLFQ (MCT < 21.59 AND Starvation < 36.0)
    ax.fill_between([28, 36], [14, 14], [21.59, 21.59],
                    color='limegreen', alpha=0.15, zorder=0)
    ax.text(29.2, 17.5, "Dominates\nMLFQ", color='darkgreen',
            fontsize=9.5, fontstyle='italic', va='center')

    # W14-ω Pareto curve
    ax.plot(starvation, mct, 'b-o', linewidth=1.8, markersize=5,
            label='W14-ω (ω sweep)', zorder=3)

    # Label each omega point
    offsets = {
        0.0: (-0.6, 0.12), 0.1: (-0.5, 0.13), 0.2: (0.3, 0.12),
        0.3: (0.3, 0.08),  0.4: (0.3, -0.18), 0.5: (-0.6, -0.20),
        0.6: (-0.7, 0.12), 0.7: (0.3, 0.10),  0.8: (0.3, -0.18),
        0.9: (-0.9, -0.18), 1.0: (-1.0, 0.12),
    }
    for om, mc, sv in zip(omega_s, mct, starvation):
        dx, dy = offsets.get(om, (0.2, 0.08))
        ax.annotate(f'ω={om:.1f}', xy=(sv, mc), xytext=(sv + dx, mc + dy),
                    fontsize=7.5, color='steelblue',
                    arrowprops=dict(arrowstyle='-', color='steelblue',
                                   lw=0.6, alpha=0.6))

    # ω=0.7 gold star
    idx07 = omega_s.index(0.7)
    ax.plot(starvation[idx07], mct[idx07], '*', color='goldenrod',
            markersize=16, zorder=5, label='ω=0.7 (recommended)')

    # Baselines
    ax.plot(mlfq_starv, mlfq_mct, 'rx', markersize=12, markeredgewidth=2.5,
            zorder=4, label='MLFQ baseline')
    ax.plot(w10c_starv, w10c_mct, 'g^', markersize=10, zorder=4,
            label='W10C (oracle)')
    ax.plot(w12_starv,  w12_mct,  's', color='gray', markersize=9, zorder=4,
            label='W12 (no burst)')

    # Reference lines for MLFQ dominance region
    ax.axhline(mlfq_mct, color='red', linewidth=0.7, linestyle=':', alpha=0.5)
    ax.axvline(mlfq_starv, color='red', linewidth=0.7, linestyle=':', alpha=0.5)

    ax.set_xlim(28, 56)
    ax.set_ylim(14, 24)
    ax.set_xlabel('Starvation Rate (%)')
    ax.set_ylabel('Mean Completion Time (s)')
    ax.set_title('W14-ω Pareto Frontier: MCT vs Starvation')
    ax.legend(loc='upper right', framealpha=0.9)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGURES_DIR, f'fig1_pareto_frontier.{ext}'),
                    bbox_inches='tight')
    plt.close(fig)
    print("Figure 1 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — W10C Learning Curve
# ─────────────────────────────────────────────────────────────────────────────

def fig2_learning_curve():
    episodes = [0, 1000, 2000, 3000, 4000, 5000,
                6000, 7000, 8000, 9000, 10000]
    mct_w10c = [21.5, 20.8, 20.1, 19.5, 19.0,
                18.8, 18.5, 18.3, 17.8, 17.5, 17.23]
    mct_rr   = 21.31
    mct_srpt = 15.20
    fresh_mean = 21.71
    fresh_std  = 0.64

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # Fresh training band
    ax.fill_between([0, 10000],
                    [fresh_mean - fresh_std] * 2,
                    [fresh_mean + fresh_std] * 2,
                    color='gray', alpha=0.18, zorder=0,
                    label=f'3-seed fresh training: {fresh_mean:.2f}±{fresh_std:.2f}s')

    # Baselines
    ax.axhline(mct_rr, color='red', linestyle='--', linewidth=1.6,
               label=f'RR ({mct_rr:.2f}s)')
    ax.axhline(mct_srpt, color='green', linestyle='--', linewidth=1.6,
               label=f'SRPT oracle ({mct_srpt:.2f}s)')

    # W10C learning curve
    ax.plot(episodes, mct_w10c, 'b-o', linewidth=2.0, markersize=5,
            label='W10C', zorder=4)

    # Gold star + annotation at final point
    ax.plot(10000, 17.23, '*', color='goldenrod', markersize=16, zorder=5)
    ax.annotate('17.23s\n(N=1500 eval)', xy=(10000, 17.23),
                xytext=(8200, 16.05),
                fontsize=9, color='goldenrod',
                arrowprops=dict(arrowstyle='->', color='goldenrod', lw=1.2))

    ax.set_xlim(0, 10000)
    ax.set_ylim(14, 24)
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Mean Completion Time (s)')
    ax.set_title('W10C Training Progression')
    ax.legend(loc='upper right', framealpha=0.9)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGURES_DIR, f'fig2_learning_curve.{ext}'),
                    bbox_inches='tight')
    plt.close(fig)
    print("Figure 2 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — W14-ω Architecture Diagram
# ─────────────────────────────────────────────────────────────────────────────

def fig3_architecture():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    def box(ax, x, y, w, h, text, color='#DDEEFF', fontsize=9,
            textcolor='black', bold=False, style='round,pad=0.1'):
        fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                                boxstyle=style,
                                facecolor=color, edgecolor='#555555',
                                linewidth=1.2, zorder=2)
        ax.add_patch(fancy)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, weight=weight,
                zorder=3, multialignment='center')

    def arrow(ax, x0, y0, x1, y1, color='#555555', lw=1.2):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=lw, connectionstyle='arc3,rad=0.0'),
                    zorder=1)

    # ── Input Layer ──────────────────────────────────────────────────────────
    box(ax, 1.2, 5.2, 1.9, 1.6,
        "Process Slots\n(5 × 7-dim)\n\ntime_in_queue\nwait_time\ncpu_norm, mem_norm\narrived_flag",
        color='#D6E8FA', fontsize=8)

    box(ax, 1.2, 2.8, 1.9, 0.9,
        "ω_starvation\n(scalar, runtime)",
        color='#FFE0B0', fontsize=8.5, bold=True)

    # ── Pre-Attention FiLM ────────────────────────────────────────────────────
    box(ax, 3.4, 5.2, 1.8, 1.0,
        "Pre-Attention FiLM\nQ = Q × (1 + ω_s)\n\nQuery Modulation",
        color='#FFE0B0', fontsize=8.5)
    ax.text(3.4, 4.45, "← shifts attn toward\n   starved tasks",
            ha='center', fontsize=7.5, color='#7a5500', style='italic')

    arrow(ax, 2.15, 5.2, 2.5, 5.2)
    arrow(ax, 1.2, 2.35, 3.4, 4.7)   # omega to pre-FiLM

    # ── 2-Head Attention ──────────────────────────────────────────────────────
    box(ax, 5.4, 5.7, 1.8, 0.75,
        "Head 1\nContext / Survey",
        color='#DDEEFF', fontsize=8.5)
    box(ax, 5.4, 4.65, 1.8, 0.75,
        "Head 2\nStarvation Monitor",
        color='#DDEEFF', fontsize=8.5)

    arrow(ax, 4.3, 5.2, 4.5, 5.7)
    arrow(ax, 4.3, 5.2, 4.5, 4.65)

    box(ax, 5.4, 3.7, 1.8, 0.7,
        "Concat + Project\ncontext (16-dim)",
        color='#C8E6C9', fontsize=8.5)

    arrow(ax, 5.4, 5.32, 5.4, 4.05)
    arrow(ax, 5.4, 4.28, 5.4, 4.05)

    # ── Post-Attention FiLM ───────────────────────────────────────────────────
    box(ax, 7.5, 3.7, 2.1, 0.85,
        "Post-Attention FiLM\nctx × (1 + ω_s) + ω_mct",
        color='#FFE0B0', fontsize=8.5)

    arrow(ax, 6.3, 3.7, 6.45, 3.7)
    arrow(ax, 1.2, 2.35, 7.5, 3.28)   # omega to post-FiLM

    # ── MLP per candidate ─────────────────────────────────────────────────────
    box(ax, 7.5, 2.35, 2.1, 0.85,
        "MLP (per candidate)\n24 → 64 → 32 → 1",
        color='#DDEEFF', fontsize=8.5)
    ax.text(7.5, 1.82,
            "[enc(7) ‖ ctx(16) ‖ ω_s(1)]",
            ha='center', fontsize=7.5, color='#1a3a6b', style='italic')

    arrow(ax, 7.5, 3.27, 7.5, 2.78)

    # ── Output ────────────────────────────────────────────────────────────────
    box(ax, 7.5, 1.1, 2.1, 0.75,
        "Q-values (15 actions)\n5 tasks × 3 quanta",
        color='#C8E6C9', fontsize=8.5, bold=True)

    arrow(ax, 7.5, 1.93, 7.5, 1.48)

    # ── Runtime control annotation ────────────────────────────────────────────
    box(ax, 1.2, 1.1, 1.9, 1.2,
        "Runtime Control\n\nω_s = 0.7 → balanced\nω_s = 0.0 → pure MCT\nω_s = 1.0 → pure fairness",
        color='#FFF3CD', fontsize=8, style='round,pad=0.15')

    arrow(ax, 2.15, 2.8, 2.5, 2.8)   # runtime → omega box visual conn

    # ── Column labels ─────────────────────────────────────────────────────────
    for xpos, label in [(1.2, 'Input'), (3.4, 'Pre-FiLM'),
                         (5.4, '2-Head Attention'), (7.5, 'MLP + Output')]:
        ax.text(xpos, 6.65, label, ha='center', fontsize=9,
                color='#333333', weight='bold')
        ax.plot([xpos - 0.85, xpos + 0.85], [6.45, 6.45],
                color='#aaaaaa', lw=0.8)

    ax.set_title('W14-ω Architecture: Preference-Conditioned Attention DQN',
                 fontsize=13, weight='bold', pad=14)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGURES_DIR, f'fig3_architecture.{ext}'),
                    bbox_inches='tight')
    plt.close(fig)
    print("Figure 3 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — VRFI Falsification (Example D)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_vrfi_falsification():
    # DejaVu Sans supports ✗ (U+2717) and ✓ (U+2713); override serif for this figure
    with mpl.rc_context({'font.family': 'DejaVu Sans'}):
        _fig4_inner()

def _fig4_inner():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # ── Left: Value curves ────────────────────────────────────────────────────
    d = np.linspace(0, 200, 500)
    v_steep  = np.maximum(0.2, np.exp(-d / 25))
    v_smooth = np.exp(-d / 100)

    ax1.plot(d, v_steep,  'b-', linewidth=2.2, label='Steep (τ=25s)')
    ax1.plot(d, v_smooth, 'r-', linewidth=2.2, label='Smooth (τ=100s)')

    # Shade area under each curve from 0 to 50
    d50 = d[d <= 50]
    ax1.fill_between(d50, np.maximum(0.2, np.exp(-d50/25)),  alpha=0.18, color='blue')
    ax1.fill_between(d50, np.exp(-d50/100), alpha=0.18, color='red')

    # Vertical dashed at d=50
    ax1.axvline(50, color='gray', linestyle='--', linewidth=1.2)
    ax1.text(52, 0.72, 'Steep: V=0.20 (floor)', fontsize=8.5, color='steelblue')
    ax1.text(52, 0.58, 'Smooth: V=0.61',        fontsize=8.5, color='firebrick')

    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Delay (s)')
    ax1.set_ylabel('V(d)')
    ax1.set_title('Value Curves at Equal Delay (d=50s)')
    ax1.legend(loc='upper right')

    # ── Right: Metric comparison bar chart ────────────────────────────────────
    metrics = ['Jain Index\n(JFI)', 'Slowdown\nVariance (SDV)', 'VRFI\n(proposed)']
    values  = [1.000, 0.000, 0.641]
    colors  = ['#E57373', '#E57373', '#66BB6A']

    bars = ax2.barh(metrics, values, color=colors, height=0.45,
                    edgecolor='#444444', linewidth=0.9)

    ax2.set_xlim(0, 1.3)
    ax2.set_xlabel('Metric Value')
    ax2.set_title('Existing Metrics Miss Value-Rate Inequality')

    # Annotations on bars
    ax2.text(1.005, 2, 'Reports: FAIR \u2717', va='center', fontsize=9.5,
             color='darkred', weight='bold')
    ax2.text(1.005, 1, 'Reports: FAIR \u2717', va='center', fontsize=9.5,
             color='darkred', weight='bold')
    ax2.text(0.655, 0, 'Detects: UNFAIR \u2713', va='center', fontsize=9.5,
             color='darkgreen', weight='bold')

    # Text box explanation
    explanation = (
        "All 10 tasks wait identical time (d=50s)\n"
        "JFI = 1.0  (perfect equality of delays)\n"
        "SDV = 0.0  (zero slowdown variance)\n"
        "Yet steep tasks lose value 2× faster than smooth"
    )
    ax2.text(0.02, -0.72, explanation, fontsize=8.2, va='center',
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFDE7',
                       edgecolor='#BDBDBD', alpha=0.95))

    fig.suptitle('VRFI Falsification: Identical Delays, Unequal Value Loss',
                 fontsize=13, weight='bold', y=1.01)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGURES_DIR, f'fig4_vrfi_falsification.{ext}'),
                    bbox_inches='tight')
    plt.close(fig)
    print("Figure 4 saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    fig1_pareto_frontier()
    fig2_learning_curve()
    fig3_architecture()
    fig4_vrfi_falsification()

    print("\nFile sizes:")
    for fname in sorted(os.listdir(FIGURES_DIR)):
        fpath = os.path.join(FIGURES_DIR, fname)
        size  = os.path.getsize(fpath)
        print(f"  {fname}: {size/1024:.1f} KB")
