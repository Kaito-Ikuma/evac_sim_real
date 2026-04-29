
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

C_MAX_DEFAULT = 50.0
EPS = 1e-12


def extract_frame_number(path: str):
    m = re.search(r"agents_frame_(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None


def find_latest_agents_csv(results_dir: str):
    files = sorted(glob.glob(os.path.join(results_dir, "agents_frame_*.csv")))
    if not files:
        raise FileNotFoundError(f"agents_frame_*.csv が見つかりません: {results_dir}")
    pairs = [(extract_frame_number(f), f) for f in files]
    pairs = [p for p in pairs if p[0] is not None]
    pairs.sort(key=lambda x: x[0])
    return pairs[-1][1]


def safe_corr(x, y, method="pearson"):
    tmp = pd.concat([pd.Series(x), pd.Series(y)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 2:
        return np.nan
    return tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method=method)


def build_tau_table(df, c_max: float):
    out = df.copy()
    out['tau_move'] = np.where(
        (out['step_when_decided'] >= 0) & (out['evacuation_time'] >= 0),
        out['evacuation_time'] - out['step_when_decided'],
        np.nan,
    )
    out['capacity_factor_post'] = 1.0 - (out['post_mean_density'] / c_max)
    out['capacity_factor_post_clipped'] = out['capacity_factor_post'].clip(lower=EPS)
    out['tau_pred_raw'] = (4.0 * out['V_at_decision']) / (
        out['post_g_down_mean'] * out['capacity_factor_post_clipped']
    )
    out['tau_pred_skeleton'] = out['V_at_decision'] / (
        out['post_g_down_mean'] * out['capacity_factor_post_clipped']
    )
    valid_fit = out[['tau_move', 'tau_pred_skeleton']].replace([np.inf, -np.inf], np.nan).dropna()
    valid_fit = valid_fit[(valid_fit['tau_move'] > 0) & (valid_fit['tau_pred_skeleton'] > 0)]
    if len(valid_fit) >= 2:
        a_hat = float((valid_fit['tau_pred_skeleton'] * valid_fit['tau_move']).sum() / (valid_fit['tau_pred_skeleton'] ** 2).sum())
        log_ratio = np.log(valid_fit['tau_move']) - np.log(valid_fit['tau_pred_skeleton'])
        a_geom = float(np.exp(log_ratio.mean()))
    else:
        a_hat = np.nan
        a_geom = np.nan
    out['tau_pred_fit_l2'] = a_hat * out['tau_pred_skeleton']
    out['tau_pred_fit_geom'] = a_geom * out['tau_pred_skeleton']
    return out, a_hat, a_geom


def make_tau_plots(df_valid, output_dir):
    pred_cols = ['tau_pred_raw', 'tau_pred_fit_l2', 'tau_pred_fit_geom']
    labels = {
        'tau_pred_raw': r'Raw $\frac{4V_i^{dec}}{\bar g_i^{post}(1-\bar\rho_i^{post}/C_{max})}$',
        'tau_pred_fit_l2': r'Fitted $a\frac{V_i^{dec}}{\bar g_i^{post}(1-\bar\rho_i^{post}/C_{max})}$ (L2)',
        'tau_pred_fit_geom': r'Fitted $a\frac{V_i^{dec}}{\bar g_i^{post}(1-\bar\rho_i^{post}/C_{max})}$ (geom)',
    }
    for col in pred_cols:
        plot_df = df_valid[[col, 'tau_move']].replace([np.inf, -np.inf], np.nan).dropna()
        if len(plot_df) == 0:
            continue
        xmax = max(plot_df[col].max(), plot_df['tau_move'].max())
        xmin = min(plot_df[col].min(), plot_df['tau_move'].min())
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(plot_df[col], plot_df['tau_move'], alpha=0.45, s=16)
        ax.plot([xmin, xmax], [xmin, xmax], '--')
        ax.set_xlabel(labels[col])
        ax.set_ylabel(r'Observed $\tau_i^{move}$')
        ax.set_title(f'tau_move validation: {col}')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{col}_validation_linear.png'), dpi=180)
        plt.close(fig)

        plot_df = plot_df[(plot_df[col] > 0) & (plot_df['tau_move'] > 0)]
        if len(plot_df) > 0:
            xmin = min(plot_df[col].min(), plot_df['tau_move'].min())
            xmax = max(plot_df[col].max(), plot_df['tau_move'].max())
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(plot_df[col], plot_df['tau_move'], alpha=0.45, s=16)
            ax.plot([xmin, xmax], [xmin, xmax], '--')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(labels[col])
            ax.set_ylabel(r'Observed $\tau_i^{move}$')
            ax.set_title(f'tau_move validation (log-log): {col}')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{col}_validation_loglog.png'), dpi=180)
            plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].scatter(df_valid['V_at_decision'], df_valid['tau_move'], alpha=0.45, s=16)
    axes[0].set_xlabel('V_at_decision')
    axes[0].set_ylabel('tau_move')
    axes[0].set_title('Distance vs tau_move')

    axes[1].scatter(df_valid['post_g_down_mean'], df_valid['tau_move'], alpha=0.45, s=16)
    axes[1].set_xlabel('post_g_down_mean')
    axes[1].set_ylabel('tau_move')
    axes[1].set_title('Post downhill freedom vs tau_move')

    axes[2].scatter(df_valid['post_mean_density'], df_valid['tau_move'], alpha=0.45, s=16)
    axes[2].set_xlabel('post_mean_density')
    axes[2].set_ylabel('tau_move')
    axes[2].set_title('Post density vs tau_move')

    axes[3].scatter(df_valid['post_no_progress_frames'], df_valid['tau_move'], alpha=0.45, s=16, label='no_progress')
    axes[3].scatter(df_valid['post_stagnated_frames'], df_valid['tau_move'], alpha=0.35, s=16, label='stagnated')
    axes[3].set_xlabel('post_*_frames')
    axes[3].set_ylabel('tau_move')
    axes[3].set_title('No-progress / stagnation vs tau_move')
    axes[3].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'tau_move_components_postmean.png'), dpi=180)
    plt.close(fig)


def analyze_exit_events(results_dir, macro_df, output_dir):
    exit_path = os.path.join(results_dir, 'exit_events.csv')
    if not os.path.exists(exit_path):
        return {}
    exit_df = pd.read_csv(exit_path).sort_values('abs_step').reset_index(drop=True)
    exit_df['headway_steps'] = exit_df['abs_step'].diff()
    exit_df.to_csv(os.path.join(output_dir, 'exit_events_with_headway.csv'), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = exit_df['headway_steps'].dropna()
    if len(vals) > 0:
        ax.hist(vals, bins=50)
    ax.set_xlabel('headway_steps')
    ax.set_ylabel('count')
    ax.set_title('Headway distribution')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'headway_distribution_steps.png'), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(exit_df['abs_step'], np.arange(1, len(exit_df) + 1))
    ax.set_xlabel('abs_step')
    ax.set_ylabel('cumulative exits')
    ax.set_title('Cumulative exit count')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cumulative_exit_count.png'), dpi=180)
    plt.close(fig)

    event_frame_counts = exit_df.groupby('frame').size().rename('service_rate_event_frame')
    macro2 = macro_df.merge(event_frame_counts, how='left', left_on='frame', right_index=True)
    macro2['service_rate_event_frame'] = macro2['service_rate_event_frame'].fillna(0)
    macro2.to_csv(os.path.join(output_dir, 'macro_stats_with_service_rate.csv'), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(macro2['h_ext'], macro2['J_out_frame'], label='J_out_frame (macro)')
    ax.plot(macro2['h_ext'], macro2['service_rate_event_frame'], label='exit_events per frame', alpha=0.8)
    ax.set_xlabel('h_ext')
    ax.set_ylabel('exits / frame')
    ax.set_title('Service rate vs h')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'service_rate_vs_h.png'), dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(macro2['h_ext'], macro2['J_out_per_step'], label='J_out_per_step')
    ax1.set_xlabel('h_ext')
    ax1.set_ylabel('J_out_per_step')
    ax2 = ax1.twinx()
    ax2.plot(macro2['h_ext'], macro2['Q_transport'], label='Q_transport')
    ax2.set_ylabel('Q_transport')
    ax1.set_title('Service rate and transport inventory')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'service_rate_and_queue.png'), dpi=180)
    plt.close(fig)

    return {
        'n_events': len(exit_df),
        'mean_headway_steps': float(vals.mean()) if len(vals) else np.nan,
        'median_headway_steps': float(vals.median()) if len(vals) else np.nan,
        'std_headway_steps': float(vals.std()) if len(vals) else np.nan,
    }


def analyze_social_trace(results_dir, macro_df, output_dir):
    trace_path = os.path.join(results_dir, 'social_trace_sample.csv')
    if not os.path.exists(trace_path):
        return {}
    trace = pd.read_csv(trace_path).sort_values(['agent_id', 'abs_step']).reset_index(drop=True)
    peak_idx = int(macro_df['A_dec_frame'].idxmax())
    peak_frame = int(macro_df.loc[peak_idx, 'frame'])
    peak_h = float(macro_df.loc[peak_idx, 'h_ext'])
    peak_A = float(macro_df.loc[peak_idx, 'A_dec_frame'])
    window = range(max(0, peak_frame - 8), min(int(macro_df['frame'].max()) + 1, peak_frame + 9))

    summ_rows = []
    aligned_rows = []
    for aid, g in trace.groupby('agent_id'):
        g = g.sort_values('abs_step').reset_index(drop=True)
        dec = g[g['tag'] == 'decision']
        if len(dec) == 0:
            continue
        dstep = int(dec['abs_step'].iloc[0])
        dframe = int(dec['frame'].iloc[0])
        dh = float(dec['h_ext'].iloc[0])
        sub = g.copy()
        sub['rel_step_from_decision'] = sub['abs_step'] - dstep
        pre = sub[sub['rel_step_from_decision'] < 0]
        post = sub[sub['rel_step_from_decision'] >= 0]
        slope_I = np.nan
        slope_H = np.nan
        if len(pre) >= 2:
            slope_I = float(np.polyfit(pre['rel_step_from_decision'], pre['I_soc'], 1)[0])
            slope_H = float(np.polyfit(pre['rel_step_from_decision'], pre['H_i'], 1)[0])
        summ_rows.append({
            'agent_id': int(aid),
            'decision_abs_step': dstep,
            'decision_frame': dframe,
            'decision_h': dh,
            'I_soc_at_decision': float(dec['I_soc'].iloc[0]),
            'H_at_decision': float(dec['H_i'].iloc[0]),
            'pre_I_soc_mean': float(pre['I_soc'].mean()) if len(pre) else np.nan,
            'pre_I_soc_std': float(pre['I_soc'].std()) if len(pre) else np.nan,
            'pre_H_mean': float(pre['H_i'].mean()) if len(pre) else np.nan,
            'pre_H_std': float(pre['H_i'].std()) if len(pre) else np.nan,
            'pre_I_soc_slope_per_step': slope_I,
            'pre_H_slope_per_step': slope_H,
            'in_spinodal_window': int(dframe in window),
        })
        keep = sub[(sub['rel_step_from_decision'] >= -64) & (sub['rel_step_from_decision'] <= 8)].copy()
        aligned_rows.append(keep)
    summary = pd.DataFrame(summ_rows)
    summary.to_csv(os.path.join(output_dir, 'social_trace_decision_summary.csv'), index=False)

    if aligned_rows:
        aligned = pd.concat(aligned_rows, ignore_index=True)
        prof = aligned.groupby('rel_step_from_decision').agg(
            mean_I_soc=('I_soc', 'mean'),
            mean_H=('H_i', 'mean'),
            n=('agent_id', 'count')
        ).reset_index()
        prof.to_csv(os.path.join(output_dir, 'aligned_social_profile.csv'), index=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(prof['rel_step_from_decision'], prof['mean_I_soc'], label='mean I_soc')
        ax.plot(prof['rel_step_from_decision'], prof['mean_H'], label='mean H')
        ax.axvline(0, linestyle='--')
        ax.set_xlabel('step - decision_step')
        ax.set_ylabel('mean value')
        ax.set_title('Aligned social growth around decision')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'aligned_social_growth.png'), dpi=180)
        plt.close(fig)

        pivot = aligned.pivot_table(index='agent_id', columns='rel_step_from_decision', values='I_soc', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(pivot.values, aspect='auto', origin='lower')
        fig.colorbar(im, ax=ax, label='I_soc')
        ax.set_xlabel('rel_step_from_decision (column index order)')
        ax.set_ylabel('sampled agent index')
        ax.set_title('I_soc heatmap around decision')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'I_soc_heatmap_around_decision.png'), dpi=180)
        plt.close(fig)

    spinodal = trace[trace['frame'].isin(list(window))]
    fig, ax = plt.subplots(figsize=(10, 6))
    for aid, g in spinodal.groupby('agent_id'):
        if len(g) > 1:
            ax.plot(g['abs_step'], g['I_soc'], alpha=0.4)
    ax.set_xlabel('abs_step')
    ax.set_ylabel('I_soc')
    ax.set_title('I_soc traces in spinodal-like window')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'I_soc_spinodal_window_traces.png'), dpi=180)
    plt.close(fig)

    return {
        'n_trace_rows': len(trace),
        'n_agents': int(trace['agent_id'].nunique()),
        'peak_frame': peak_frame,
        'peak_h': peak_h,
        'peak_A_dec_frame': peak_A,
        'corr_I_H': safe_corr(trace['I_soc'], trace['H_i']),
        'window_frames': list(window),
    }


def main():
    parser = argparse.ArgumentParser(description='PriorityC post-mean validation analysis')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='priorityC_postmean_validation')
    parser.add_argument('--agents_csv', type=str, default=None)
    parser.add_argument('--c_max', type=float, default=C_MAX_DEFAULT)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    agents_csv = args.agents_csv or find_latest_agents_csv(args.results_dir)
    df = pd.read_csv(agents_csv)
    macro_path = os.path.join(args.results_dir, 'macro_stats.csv')
    macro_df = pd.read_csv(macro_path)

    tau_df, a_hat, a_geom = build_tau_table(df, args.c_max)
    tau_valid = tau_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['tau_move', 'tau_pred_raw', 'tau_pred_fit_l2', 'tau_pred_fit_geom'])
    tau_valid.to_csv(os.path.join(args.output_dir, 'tau_move_validation_table_postmean.csv'), index=False)
    make_tau_plots(tau_valid, args.output_dir)

    summary_lines = []
    summary_lines.append('=== tau_move post-mean formula validation ===')
    summary_lines.append(f'n_valid: {len(tau_valid)}')
    summary_lines.append(f'a_hat_l2: {a_hat}')
    summary_lines.append(f'a_hat_geom: {a_geom}')
    for col in ['tau_pred_raw', 'tau_pred_fit_l2', 'tau_pred_fit_geom']:
        summary_lines.append(f'pearson_{col}: {safe_corr(tau_valid[col], tau_valid["tau_move"], "pearson")}')
        summary_lines.append(f'spearman_{col}: {safe_corr(tau_valid[col], tau_valid["tau_move"], "spearman")}')
        pos = tau_valid[(tau_valid[col] > 0) & (tau_valid['tau_move'] > 0)]
        summary_lines.append(f'pearson_log_{col}: {safe_corr(np.log(pos[col]), np.log(pos["tau_move"]), "pearson")}')
        summary_lines.append(f'spearman_log_{col}: {safe_corr(np.log(pos[col]), np.log(pos["tau_move"]), "spearman")}')
    summary_lines.append(f'corr_V_tau: {safe_corr(tau_valid["V_at_decision"], tau_valid["tau_move"])}')
    summary_lines.append(f'corr_post_g_tau: {safe_corr(tau_valid["post_g_down_mean"], tau_valid["tau_move"])}')
    summary_lines.append(f'corr_post_rho_tau: {safe_corr(tau_valid["post_mean_density"], tau_valid["tau_move"])}')
    summary_lines.append(f'corr_no_progress_tau: {safe_corr(tau_valid["post_no_progress_frames"], tau_valid["tau_move"])}')
    summary_lines.append(f'corr_stagnated_tau: {safe_corr(tau_valid["post_stagnated_frames"], tau_valid["tau_move"])}')
    summary_lines.append('')

    exit_stats = analyze_exit_events(args.results_dir, macro_df, args.output_dir)
    summary_lines.append('=== exit/headway/service-rate ===')
    for k, v in exit_stats.items():
        summary_lines.append(f'{k}: {v}')
    summary_lines.append('')

    social_stats = analyze_social_trace(args.results_dir, macro_df, args.output_dir)
    summary_lines.append('=== social-trace / avalanche ===')
    for k, v in social_stats.items():
        summary_lines.append(f'{k}: {v}')
    summary_lines.append('')

    outtxt = os.path.join(args.output_dir, 'analysis_summary_postmean.txt')
    with open(outtxt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    print('\n'.join(summary_lines))


if __name__ == '__main__':
    main()
