"""
Generate training plots from WandB runs for README and BLOG embedding.

Pulls data from the heat-treatment-grpo WandB project and creates
publication-quality plots saved as PNGs in docs/plots/.

Usage:
    WANDB_API_KEY="..." python docs/generate_plots.py
"""

import os
import wandb
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
PROJECT = "mukundnjoy-paypal/heat-treatment-grpo"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.dpi': 150,
})

ACCENT = '#58a6ff'
ACCENT2 = '#f78166'
ACCENT3 = '#3fb950'
ACCENT4 = '#d2a8ff'

api = wandb.Api()


def get_run_history(run_id, keys=None):
    """Pull full history for a run, optionally filtering to specific keys."""
    run = api.run(f"{PROJECT}/{run_id}")
    df = run.history(samples=10000)
    if keys:
        available = [k for k in keys if k in df.columns]
        df = df[['_step'] + available].dropna(subset=available, how='all')
    return df


def smooth(series, window=20):
    """Exponential moving average for smoothing noisy training curves."""
    return series.ewm(span=window, min_periods=1).mean()


# ── Identify key runs ──────────────────────────────────────────────────
# Get all runs sorted by creation time
runs = api.runs(PROJECT, order="+created_at")
run_info = []
for r in runs:
    run_info.append({
        'id': r.id,
        'name': r.name,
        'state': r.state,
        'steps': r.lastHistoryStep,
        'created': r.created_at,
    })

print(f"Found {len(run_info)} runs total")

# Pick the longest finished run and the currently running one
longest_finished = max([r for r in run_info if r['state'] == 'finished'], key=lambda x: x['steps'])
currently_running = [r for r in run_info if r['state'] == 'running']
latest_run = currently_running[0] if currently_running else run_info[-1]

print(f"Longest finished: {longest_finished['id']} ({longest_finished['steps']} steps)")
print(f"Latest/running: {latest_run['id']} ({latest_run['steps']} steps)")

# ── Collect data from multiple runs to show full training arc ──────────
# We'll concatenate runs in chronological order to show the full training journey
PHYSICS_KEYS = [
    'physics/reward', 'physics/radius_nm', 'physics/core_temp_C',
    'physics/max_temp_C', 'physics/entered_growth', 'physics/recipe_steps'
]
TRAIN_KEYS = [
    'train/reward', 'train/loss', 'train/kl', 'train/grad_norm',
    'train/global_step', 'train/completion_length',
    'train/rewards/api_physics_reward_func/mean',
    'train/rewards/api_physics_reward_func/std',
]
ALL_KEYS = PHYSICS_KEYS + TRAIN_KEYS

# Pull from the two most significant runs
print(f"\nPulling data from longest run: {longest_finished['id']}...")
df_long = get_run_history(longest_finished['id'], ALL_KEYS)
print(f"  Got {len(df_long)} rows")

print(f"Pulling data from latest run: {latest_run['id']}...")
df_latest = get_run_history(latest_run['id'], ALL_KEYS)
print(f"  Got {len(df_latest)} rows")

# Use the longer dataset as our primary source
df = df_long if len(df_long) >= len(df_latest) else df_latest
primary_id = longest_finished['id'] if len(df_long) >= len(df_latest) else latest_run['id']
print(f"\nUsing primary run: {primary_id} ({len(df)} rows)")
print(f"Available columns: {[c for c in df.columns if c != '_step']}")


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: Reward Curve (the most important plot)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

reward_col = 'train/reward' if 'train/reward' in df.columns else 'physics/reward'
if reward_col in df.columns:
    data = df[['_step', reward_col]].dropna()
    ax.scatter(data['_step'], data[reward_col], alpha=0.15, s=8, color=ACCENT, label='Per-step reward')
    ax.plot(data['_step'], smooth(data[reward_col], 30), color=ACCENT2, linewidth=2, label='Smoothed (EMA-30)')
    ax.axhline(y=0, color='#484f58', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=-154, color=ACCENT2, linestyle=':', linewidth=1.5, alpha=0.7, label='Baseline (~-154)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Reward')
    ax.set_title('GRPO Training Reward Curve — Heat Treatment Scheduler', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'reward_curve.png')
    fig.savefig(path, bbox_inches='tight')
    print(f"✓ Saved {path}")
else:
    print(f"✗ No reward column found")
plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: Radius Evolution
# ═══════════════════════════════════════════════════════════════════════
if 'physics/radius_nm' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    data = df[['_step', 'physics/radius_nm']].dropna()
    ax.scatter(data['_step'], data['physics/radius_nm'], alpha=0.2, s=10, color=ACCENT3, label='Episode final radius')
    ax.plot(data['_step'], smooth(data['physics/radius_nm'], 30), color=ACCENT2, linewidth=2, label='Smoothed (EMA-30)')
    
    # Target window
    ax.axhspan(10, 15, alpha=0.1, color=ACCENT3, label='Target window (10–15 nm)')
    ax.axhline(y=12.5, color=ACCENT3, linestyle='--', linewidth=1, alpha=0.5, label='Target center (12.5 nm)')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Precipitate Radius (nm)')
    ax.set_title('Precipitate Radius Evolution During GRPO Training', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'radius_evolution.png')
    fig.savefig(path, bbox_inches='tight')
    print(f"✓ Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PLOT 3: Temperature Control (peak temp over training)
# ═══════════════════════════════════════════════════════════════════════
temp_col = 'physics/max_temp_C' if 'physics/max_temp_C' in df.columns else 'physics/core_temp_C'
if temp_col in df.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    data = df[['_step', temp_col]].dropna()
    ax.scatter(data['_step'], data[temp_col], alpha=0.15, s=8, color=ACCENT4, label='Episode peak temperature')
    ax.plot(data['_step'], smooth(data[temp_col], 30), color=ACCENT2, linewidth=2, label='Smoothed (EMA-30)')
    
    # Growth zone markers
    ax.axhline(y=176, color=ACCENT3, linestyle='--', linewidth=1, alpha=0.6, label='Growth threshold (176°C)')
    ax.axhline(y=341, color='#f0883e', linestyle='--', linewidth=1, alpha=0.6, label='Ripening threshold (341°C)')
    ax.axhline(y=502, color='#f85149', linestyle='--', linewidth=1, alpha=0.6, label='Melting point (502°C)')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Peak Material Temperature (°C)')
    ax.set_title('Temperature Control Learning — Agent Learns to Moderate Heat', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.3, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'temperature_control.png')
    fig.savefig(path, bbox_inches='tight')
    print(f"✓ Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PLOT 4: Growth Phase Entry Rate (rolling average)
# ═══════════════════════════════════════════════════════════════════════
if 'physics/entered_growth' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 4))
    data = df[['_step', 'physics/entered_growth']].dropna()
    rolling_rate = data['physics/entered_growth'].rolling(50, min_periods=1).mean()
    ax.plot(data['_step'], rolling_rate, color=ACCENT3, linewidth=2)
    ax.fill_between(data['_step'], 0, rolling_rate, alpha=0.15, color=ACCENT3)
    ax.axhline(y=0.10, color=ACCENT2, linestyle=':', linewidth=1.5, alpha=0.7, label='Baseline rate (~10%)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Growth Phase Entry Rate (rolling 50)')
    ax.set_title('Growth Phase Discovery — From 10% (Baseline) to 90%+ (Trained)', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='lower right', framealpha=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'growth_phase_entry.png')
    fig.savefig(path, bbox_inches='tight')
    print(f"✓ Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PLOT 5: Training Loss
# ═══════════════════════════════════════════════════════════════════════
if 'train/loss' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 4))
    data = df[['_step', 'train/loss']].dropna()
    ax.plot(data['_step'], data['train/loss'], alpha=0.3, color=ACCENT, linewidth=0.8)
    ax.plot(data['_step'], smooth(data['train/loss'], 30), color=ACCENT2, linewidth=2, label='Smoothed loss')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('GRPO Training Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'training_loss.png')
    fig.savefig(path, bbox_inches='tight')
    print(f"✓ Saved {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PLOT 6: Combined Dashboard (2×2 for README hero image)
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GRPO Training Dashboard — Heat Treatment Digital Twin', fontsize=16, fontweight='bold', y=0.98)

# Top-left: Reward
ax = axes[0, 0]
if reward_col in df.columns:
    data = df[['_step', reward_col]].dropna()
    ax.scatter(data['_step'], data[reward_col], alpha=0.1, s=5, color=ACCENT)
    ax.plot(data['_step'], smooth(data[reward_col], 30), color=ACCENT2, linewidth=2)
    ax.axhline(y=0, color='#484f58', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=-154, color=ACCENT2, linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_title('Reward', fontweight='bold')
ax.set_xlabel('Training Step')
ax.set_ylabel('Reward')
ax.grid(True, alpha=0.3)

# Top-right: Radius
ax = axes[0, 1]
if 'physics/radius_nm' in df.columns:
    data = df[['_step', 'physics/radius_nm']].dropna()
    ax.scatter(data['_step'], data['physics/radius_nm'], alpha=0.15, s=5, color=ACCENT3)
    ax.plot(data['_step'], smooth(data['physics/radius_nm'], 30), color=ACCENT2, linewidth=2)
    ax.axhspan(10, 15, alpha=0.1, color=ACCENT3)
    ax.axhline(y=12.5, color=ACCENT3, linestyle='--', linewidth=1, alpha=0.5)
ax.set_title('Precipitate Radius (nm)', fontweight='bold')
ax.set_xlabel('Training Step')
ax.set_ylabel('Radius (nm)')
ax.grid(True, alpha=0.3)

# Bottom-left: Temperature
ax = axes[1, 0]
if temp_col in df.columns:
    data = df[['_step', temp_col]].dropna()
    ax.scatter(data['_step'], data[temp_col], alpha=0.1, s=5, color=ACCENT4)
    ax.plot(data['_step'], smooth(data[temp_col], 30), color=ACCENT2, linewidth=2)
    ax.axhline(y=176, color=ACCENT3, linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=502, color='#f85149', linestyle='--', linewidth=1, alpha=0.5)
ax.set_title('Peak Temperature (°C)', fontweight='bold')
ax.set_xlabel('Training Step')
ax.set_ylabel('Temperature (°C)')
ax.grid(True, alpha=0.3)

# Bottom-right: Growth entry rate
ax = axes[1, 1]
if 'physics/entered_growth' in df.columns:
    data = df[['_step', 'physics/entered_growth']].dropna()
    rolling = data['physics/entered_growth'].rolling(50, min_periods=1).mean()
    ax.plot(data['_step'], rolling, color=ACCENT3, linewidth=2)
    ax.fill_between(data['_step'], 0, rolling, alpha=0.15, color=ACCENT3)
    ax.axhline(y=0.10, color=ACCENT2, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylim(-0.05, 1.1)
ax.set_title('Growth Phase Entry Rate', fontweight='bold')
ax.set_xlabel('Training Step')
ax.set_ylabel('Rate (rolling 50)')
ax.grid(True, alpha=0.3)

fig.tight_layout()
path = os.path.join(OUTPUT_DIR, 'training_dashboard.png')
fig.savefig(path, bbox_inches='tight')
print(f"✓ Saved {path}")
plt.close()

print(f"\n✅ All plots saved to {OUTPUT_DIR}/")
print("Files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f} ({size//1024} KB)")
