import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_agents_csv(results_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(results_dir, 'agents_frame_*.csv')))
    if not files:
        raise FileNotFoundError(f'No agents_frame_*.csv found in {results_dir}')
    def frame_num(path: str) -> int:
        m = re.search(r'agents_frame_(\d+)\.csv$', os.path.basename(path))
        return int(m.group(1)) if m else -1
    files.sort(key=frame_num)
    return files[-1]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_corr(x: pd.Series, y: pd.Series, method: str = 'pearson') -> float:
    tmp = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 2:
        return np.nan
    return float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method=method))


def build_tau_move_validation(df_agents: pd.DataFrame, cmax: float) -> pd.DataFrame:
    required = ['step_when_decided', 'evacuation_time', 'V_at_decision', 'g_down_at_decision', 'density_at_decision']
    missing = [c for c in required if c not in df_agents.columns]
    if missing:
        raise ValueError(f'Missing required columns for tau_move validation: {missing}')

    out = df_agents.copy()
    out['tau_move'] = np.where(
        (out['step_when_decided'] >= 0) & (out['evacuation_time'] >= 0),
        out['evacuation_time'] - out['step_when_decided'],
        np.nan,
    )
    out['rho_ratio'] = out['density_at_decision'] / cmax
    out['capacity_factor'] = 1.0 - out['rho_ratio']
    out['geom_capacity'] = out['g_down_at_decision'] * out['capacity_factor']
    out['tau_move_pred'] = np.where(
        (out['V_at_decision'] > 0) & (out['g_down_at_decision'] > 0) & (out['capacity_factor'] > 0),
        4.0 * out['V_at_decision'] / out['geom_capacity'],
        np.nan,
    )
    out['tau_move_pred_log'] = np.log10(out['tau_move_pred'])
    out['tau_move_log'] = np.log10(out['tau_move'])
    return out


def plot_tau_move_validation(df: pd.DataFrame, outdir: str) -> dict:
    plot_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['tau_move', 'tau_move_pred']).copy()
    stats = {
        'n_valid': int(len(plot_df)),
        'pearson': safe_corr(plot_df['tau_move_pred'], plot_df['tau_move'], 'pearson'),
        'spearman': safe_corr(plot_df['tau_move_pred'], plot_df['tau_move'], 'spearman'),
        'pearson_log': safe_corr(plot_df['tau_move_pred_log'], plot_df['tau_move_log'], 'pearson'),
        'spearman_log': safe_corr(plot_df['tau_move_pred_log'], plot_df['tau_move_log'], 'spearman'),
        'corr_V_tau': safe_corr(plot_df['V_at_decision'], plot_df['tau_move'], 'pearson'),
        'corr_g_tau': safe_corr(plot_df['g_down_at_decision'], plot_df['tau_move'], 'pearson'),
        'corr_rho_tau': safe_corr(plot_df['density_at_decision'], plot_df['tau_move'], 'pearson'),
    }

    # scatter linear
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(plot_df['tau_move_pred'], plot_df['tau_move'], s=14, alpha=0.5)
    if len(plot_df) >= 2:
        lo = min(plot_df['tau_move_pred'].min(), plot_df['tau_move'].min())
        hi = max(plot_df['tau_move_pred'].max(), plot_df['tau_move'].max())
        ax.plot([lo, hi], [lo, hi], linestyle='--')
    ax.set_xlabel(r'Predicted $\tau_i^{move} \approx \frac{4V_i}{g_i(1-\rho_i/C_{max})}$')
    ax.set_ylabel(r'Observed $\tau_i^{move}$')
    ax.set_title('tau_move validation (linear scale)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'tau_move_validation_scatter.png'), dpi=200)
    plt.close(fig)

    # scatter log-log
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = plot_df[(plot_df['tau_move_pred'] > 0) & (plot_df['tau_move'] > 0)]
    ax.scatter(pos['tau_move_pred'], pos['tau_move'], s=14, alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if len(pos) >= 2:
        lo = min(pos['tau_move_pred'].min(), pos['tau_move'].min())
        hi = max(pos['tau_move_pred'].max(), pos['tau_move'].max())
        ax.plot([lo, hi], [lo, hi], linestyle='--')
    ax.set_xlabel(r'Predicted $\tau_i^{move}$')
    ax.set_ylabel(r'Observed $\tau_i^{move}$')
    ax.set_title('tau_move validation (log-log)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'tau_move_validation_loglog.png'), dpi=200)
    plt.close(fig)

    # residuals vs components
    plot_df['residual'] = plot_df['tau_move'] - plot_df['tau_move_pred']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].scatter(plot_df['V_at_decision'], plot_df['tau_move'], s=14, alpha=0.5)
    axes[0].set_xlabel('V_at_decision')
    axes[0].set_ylabel('tau_move')
    axes[0].set_title('Distance vs tau_move')
    axes[1].scatter(plot_df['g_down_at_decision'], plot_df['tau_move'], s=14, alpha=0.5)
    axes[1].set_xlabel('g_down_at_decision')
    axes[1].set_ylabel('tau_move')
    axes[1].set_title('Downhill freedom vs tau_move')
    axes[2].scatter(plot_df['density_at_decision'], plot_df['tau_move'], s=14, alpha=0.5)
    axes[2].set_xlabel('density_at_decision')
    axes[2].set_ylabel('tau_move')
    axes[2].set_title('Decision density vs tau_move')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'tau_move_components_vs_observed.png'), dpi=200)
    plt.close(fig)

    plot_df[['agent_id', 'tau_move', 'tau_move_pred', 'V_at_decision', 'g_down_at_decision', 'density_at_decision', 'rho_ratio', 'capacity_factor']].to_csv(
        os.path.join(outdir, 'tau_move_validation_table.csv'), index=False
    )
    return stats


def analyze_exit_events(df_exit: pd.DataFrame, df_macro: pd.DataFrame, outdir: str) -> dict:
    if df_exit.empty:
        return {'n_events': 0}
    ex = df_exit.sort_values('abs_step').reset_index(drop=True).copy()
    ex['headway_steps'] = ex['abs_step'].diff()
    ex['headway_frames'] = ex['frame'].diff()

    stats = {
        'n_events': int(len(ex)),
        'mean_headway_steps': float(ex['headway_steps'].dropna().mean()),
        'median_headway_steps': float(ex['headway_steps'].dropna().median()),
        'std_headway_steps': float(ex['headway_steps'].dropna().std()),
    }

    ex.to_csv(os.path.join(outdir, 'exit_events_with_headway.csv'), index=False)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    hw = ex['headway_steps'].dropna()
    ax.hist(hw, bins=min(60, max(10, int(np.sqrt(len(hw))))))
    ax.set_xlabel('Headway in abs_step')
    ax.set_ylabel('Count')
    ax.set_title('Headway distribution of exit events')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'headway_distribution_steps.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(ex['abs_step'], np.arange(1, len(ex) + 1))
    ax.set_xlabel('abs_step')
    ax.set_ylabel('Cumulative exits')
    ax.set_title('Cumulative exit count')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'cumulative_exit_count.png'), dpi=200)
    plt.close(fig)

    # service rate from macro and from direct event histogram
    macro = df_macro.copy()
    macro['service_rate_event_frame'] = macro['frame'].map(ex.groupby('frame').size()).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(macro['h_ext'], macro['J_out_frame'], label='J_out_frame (macro)')
    ax.plot(macro['h_ext'], macro['service_rate_event_frame'], label='exit count per frame (events)', alpha=0.8)
    ax.set_xlabel('h_ext')
    ax.set_ylabel('Exits per frame')
    ax.set_title('Service rate vs h_ext')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'service_rate_vs_h.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(macro['frame'], macro['J_out_per_step'], label='J_out_per_step')
    ax.plot(macro['frame'], macro['Q_transport'], label='Q_transport')
    ax.set_xlabel('frame')
    ax.set_ylabel('Rate / queue')
    ax.set_title('Service rate and transport queue')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'service_rate_and_queue.png'), dpi=200)
    plt.close(fig)

    macro.to_csv(os.path.join(outdir, 'macro_stats_with_service_rate.csv'), index=False)
    return stats


def detect_spinodal_window(df_macro: pd.DataFrame, half_window: int = 8) -> dict:
    macro = df_macro.copy()
    idx = int(macro['A_dec_frame'].idxmax())
    peak_row = macro.loc[idx]
    frame0 = int(peak_row['frame'])
    mask = (macro['frame'] >= frame0 - half_window) & (macro['frame'] <= frame0 + half_window)
    return {
        'peak_frame': frame0,
        'peak_h': float(peak_row['h_ext']),
        'peak_A_dec_frame': float(peak_row['A_dec_frame']),
        'window_frames': macro.loc[mask, 'frame'].tolist(),
    }


def analyze_social_trace(df_trace: pd.DataFrame, df_macro: pd.DataFrame, outdir: str) -> dict:
    if df_trace.empty:
        return {'n_trace_rows': 0}

    trace = df_trace.sort_values(['agent_id', 'abs_step']).reset_index(drop=True).copy()
    spin = detect_spinodal_window(df_macro)
    window_frames = set(spin['window_frames'])
    trace['in_spinodal_window'] = trace['frame'].isin(window_frames)

    stats = {
        'n_trace_rows': int(len(trace)),
        'n_agents': int(trace['agent_id'].nunique()),
        'peak_frame': spin['peak_frame'],
        'peak_h': spin['peak_h'],
        'peak_A_dec_frame': spin['peak_A_dec_frame'],
        'corr_I_H': safe_corr(trace['I_soc'], trace['H_i'], 'pearson'),
    }

    # decision-time summary per sampled agent
    records = []
    for aid, g in trace.groupby('agent_id'):
        g = g.sort_values('abs_step').reset_index(drop=True)
        dec = g[g['tag'] == 'decision']
        if len(dec) == 0:
            continue
        dec_row = dec.iloc[0]
        pre = g[g['tag'] == 'pre_window']
        records.append({
            'agent_id': aid,
            'decision_abs_step': int(dec_row['abs_step']),
            'decision_frame': int(dec_row['frame']),
            'decision_h': float(dec_row['h_ext']),
            'I_soc_at_decision': float(dec_row['I_soc']),
            'H_at_decision': float(dec_row['H_i']),
            'pre_I_soc_mean': float(pre['I_soc'].mean()) if len(pre) else np.nan,
            'pre_I_soc_std': float(pre['I_soc'].std()) if len(pre) else np.nan,
            'pre_H_mean': float(pre['H_i'].mean()) if len(pre) else np.nan,
            'pre_H_std': float(pre['H_i'].std()) if len(pre) else np.nan,
            'pre_I_soc_slope_per_step': float(np.polyfit(pre['abs_step'], pre['I_soc'], 1)[0]) if len(pre) >= 2 else np.nan,
            'pre_H_slope_per_step': float(np.polyfit(pre['abs_step'], pre['H_i'], 1)[0]) if len(pre) >= 2 else np.nan,
            'in_spinodal_window': bool(int(dec_row['frame']) in window_frames),
        })
    summary = pd.DataFrame(records)
    summary.to_csv(os.path.join(outdir, 'social_trace_decision_summary.csv'), index=False)

    # plot all sampled agents aligned to decision
    aligned_rows = []
    for aid, g in trace.groupby('agent_id'):
        g = g.sort_values('abs_step').reset_index(drop=True)
        dec = g[g['tag'] == 'decision']
        if len(dec) == 0:
            continue
        t0 = int(dec.iloc[0]['abs_step'])
        gg = g.copy()
        gg['rel_step'] = gg['abs_step'] - t0
        aligned_rows.append(gg)
    aligned = pd.concat(aligned_rows, ignore_index=True) if aligned_rows else pd.DataFrame(columns=trace.columns)

    if len(aligned):
        prof = aligned.groupby('rel_step').agg(
            mean_I_soc=('I_soc', 'mean'),
            mean_H=('H_i', 'mean'),
            n=('agent_id', 'count')
        ).reset_index()
        prof.to_csv(os.path.join(outdir, 'aligned_social_profile.csv'), index=False)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(prof['rel_step'], prof['mean_I_soc'], label='mean I_soc')
        ax.plot(prof['rel_step'], prof['mean_H'], label='mean H_i')
        ax.axvline(0, linestyle='--')
        ax.set_xlabel('abs_step - decision_abs_step')
        ax.set_ylabel('Mean field')
        ax.set_title('Aligned social-field growth around decision')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'aligned_social_growth.png'), dpi=200)
        plt.close(fig)

        pivot = aligned.pivot_table(index='agent_id', columns='rel_step', values='I_soc', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.to_numpy(), aspect='auto', interpolation='nearest', origin='lower')
        ax.set_xlabel('relative step index')
        ax.set_ylabel('sampled agent index')
        ax.set_title('Heatmap of I_soc around decision')
        fig.colorbar(im, ax=ax, label='I_soc')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'I_soc_heatmap_around_decision.png'), dpi=200)
        plt.close(fig)

    # spinodal-window subset
    spin_trace = trace[trace['in_spinodal_window']].copy()
    if len(spin_trace):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for aid, g in spin_trace.groupby('agent_id'):
            ax.plot(g['abs_step'], g['I_soc'], alpha=0.2)
        ax.set_xlabel('abs_step')
        ax.set_ylabel('I_soc')
        ax.set_title('Sampled I_soc traces in spinodal window')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'I_soc_spinodal_window_traces.png'), dpi=200)
        plt.close(fig)

    return stats


def write_summary(summary_path: str, tau_stats: dict, exit_stats: dict, social_stats: dict, spin: dict) -> None:
    lines = []
    lines.append('=== tau_move formula validation ===')
    for k, v in tau_stats.items():
        lines.append(f'{k}: {v}')
    lines.append('')
    lines.append('=== exit/headway/service-rate ===')
    for k, v in exit_stats.items():
        lines.append(f'{k}: {v}')
    lines.append('')
    lines.append('=== social-trace / avalanche ===')
    for k, v in social_stats.items():
        lines.append(f'{k}: {v}')
    lines.append('')
    lines.append('=== detected spinodal-like window from A_dec peak ===')
    for k, v in spin.items():
        lines.append(f'{k}: {v}')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    

    parser = argparse.ArgumentParser(description='Analyze Priority-C outputs: tau_move formula, headway/service rate, and social-field avalanche traces.')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing agents_frame_*.csv and macro/exit/social outputs')
    parser.add_argument('--agents_csv', type=str, default=None, help='Optional explicit agents_frame_*.csv path')
    parser.add_argument('--macro_csv', type=str, default=None, help='Optional macro_stats.csv path')
    parser.add_argument('--exit_csv', type=str, default=None, help='Optional exit_events.csv path')
    parser.add_argument('--trace_csv', type=str, default=None, help='Optional social_trace_sample.csv path')
    parser.add_argument('--output_dir', type=str, default='priorityC_validation_results', help='Output directory')
    parser.add_argument('--cmax', type=float, default=50.0, help='Capacity C_max used in the simulation')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    agents_csv = args.agents_csv or find_latest_agents_csv(args.results_dir)
    macro_csv = args.macro_csv or os.path.join(args.results_dir, 'macro_stats.csv')
    exit_csv = args.exit_csv or os.path.join(args.results_dir, 'exit_events.csv')
    trace_csv = args.trace_csv or os.path.join(args.results_dir, 'social_trace_sample.csv')

    

    df_agents = pd.read_csv(agents_csv)
    df_macro = pd.read_csv(macro_csv)
    df_exit = pd.read_csv(exit_csv)
    df_trace = pd.read_csv(trace_csv)

    

    tau_df = build_tau_move_validation(df_agents, args.cmax)
    tau_stats = plot_tau_move_validation(tau_df, args.output_dir)
    exit_stats = analyze_exit_events(df_exit, df_macro, args.output_dir)
    social_stats = analyze_social_trace(df_trace, df_macro, args.output_dir)
    spin = detect_spinodal_window(df_macro)
    write_summary(os.path.join(args.output_dir, 'analysis_summary.txt'), tau_stats, exit_stats, social_stats, spin)

    print('[Saved files]')
    for f in sorted(glob.glob(os.path.join(args.output_dir, '*'))):
        print(' -', f)


if __name__ == '__main__':
    main()
