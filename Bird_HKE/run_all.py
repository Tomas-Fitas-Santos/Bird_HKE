"""
Run main.py with every combination of (config × video+gt) defined in run_config.json.

Usage:
    python Bird_HKE/run_all.py                         # uses Bird_HKE/experiments/run_config.json
    python Bird_HKE/run_all.py --config my_config.json  # custom config path
    python Bird_HKE/run_all.py --dry-run                # validate all paths & print commands
"""
import json
import sys
import os
import csv
import subprocess
import argparse
import itertools
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


# ── helpers ──────────────────────────────────────────────────────────────────

_SUMMARY_KEYS = [
    'config', 'video', 'flags', 'params_M', 'gflops',
    'initial_PCK@0.05_mean', 'initial_PCK@0.05_visible', 'initial_PCK@0.05_occluded',
    'initial_PCK@0.05_threshold_px',
    'initial_normalized_jitter', 'initial_velocity_error', 'initial_acceleration_error',
    'final_PCK@0.05_mean',   'final_PCK@0.05_visible',   'final_PCK@0.05_occluded',
    'final_PCK@0.05_threshold_px',
    'final_normalized_jitter', 'final_velocity_error',   'final_acceleration_error',
]


def _get_output_dir_from_yaml(yaml_path):
    """Return the OUTPUT_DIR value from a YAML config, or '' if unavailable."""
    if yaml is None:
        return ''
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        # TEST.OUTPUT_DIR takes precedence (mirrors default.py logic)
        out = ''
        try:
            out = data['TEST']['OUTPUT_DIR']
        except (KeyError, TypeError):
            pass
        if not out:
            try:
                out = data['OUTPUT_DIR']
            except (KeyError, TypeError):
                pass
        return (out or '').strip("'\"")
    except Exception:
        return ''


def _parse_metrics_file(path):
    """Parse evaluation_metrics.txt into {key: float}, silently ignoring missing file."""
    result = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('='):
                    continue
                if ':' in line:
                    k, _, v = line.partition(':')
                    try:
                        result[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
    except Exception:
        pass
    return result


def _collect_run_results(yaml_path, vid, flags):
    """Read metrics + model stats written by main.py and return a summary row dict."""
    row = {
        'config':  Path(yaml_path).stem,
        'video':   Path(vid['video']).stem,
        'flags':   ' '.join(flags),
        'params_M': '',
        'gflops':   '',
    }
    for k in _SUMMARY_KEYS:
        if k not in row:
            row[k] = ''

    out_dir = _get_output_dir_from_yaml(yaml_path)
    if not out_dir:
        return row

    results_dir = Path(out_dir) / Path(vid['video']).stem

    # Model stats
    stats_path = results_dir / 'model_stats.json'
    if stats_path.exists():
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            row['params_M'] = stats.get('params_M', '')
            row['gflops']   = stats.get('gflops', '')
        except Exception:
            pass

    # Metric keys as written by metrics.py
    _kmap = [
        ('PCK@0.05_mean',          'PCK@0.05_mean'),
        ('PCK@0.05_visible',       'PCK@0.05_visible_mean'),
        ('PCK@0.05_occluded',      'PCK@0.05_occluded_mean'),
        ('PCK@0.05_threshold_px',  'PCK@0.05_threshold_px'),
        ('normalized_jitter',      'normalized_jitter'),
        ('velocity_error',         'velocity_error'),
        ('acceleration_error',     'acceleration_error'),
    ]
    for split in ('initial', 'final'):
        m = _parse_metrics_file(results_dir / split / 'evaluation_metrics.txt')
        for col_suffix, file_key in _kmap:
            row[f'{split}_{col_suffix}'] = m.get(file_key, '')

    return row


def _write_summary(summary_path, rows):
    """Write (or overwrite) the CSV summary file with all collected rows."""
    try:
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=_SUMMARY_KEYS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        print(f'Warning: could not write summary file: {exc}')


def _check_path(path_str, label, errors, warnings, is_warning=False):
    """Append to errors/warnings if a path doesn't exist."""
    if not path_str:
        return
    p = Path(path_str)
    if not p.exists():
        msg = f"  {label}: {path_str}"
        if is_warning:
            warnings.append(msg)
        else:
            errors.append(msg)


def _parse_yaml_paths(yaml_path):
    """Extract key paths from a YAML config that must/should exist."""
    if yaml is None:
        return {}, ["  (install PyYAML to enable YAML validation: pip install pyyaml)"]

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    paths = {}
    # TEST.POSE_MODEL_FILE  (critical – the trained weights)
    try:
        paths["TEST.POSE_MODEL_FILE"] = data["TEST"]["POSE_MODEL_FILE"]
    except (KeyError, TypeError):
        pass

    # OUTPUT_DIR
    try:
        paths["OUTPUT_DIR"] = data["OUTPUT_DIR"]
    except (KeyError, TypeError):
        pass

    # TEST.OUTPUT_DIR (some configs nest it)
    try:
        paths["TEST.OUTPUT_DIR"] = data["TEST"]["OUTPUT_DIR"]
    except (KeyError, TypeError):
        pass

    return paths, []


def validate_all(configs, videos, flags):
    """Check every file/dir referenced in the run matrix. Returns (errors, warnings)."""
    errors = []
    warnings = []

    # 1. main.py itself
    _check_path("Bird_HKE/main.py", "main.py script", errors, warnings)

    # 2. YAML configs
    for cfg in configs:
        _check_path(cfg, f"config", errors, warnings)

        if Path(cfg).exists():
            yaml_paths, yaml_warns = _parse_yaml_paths(cfg)
            warnings.extend(yaml_warns)

            cfg_stem = Path(cfg).stem
            for key, val in yaml_paths.items():
                if not val or val.startswith("r''") or val == "''":
                    continue
                val_clean = val.strip("'\"")
                if key == "TEST.POSE_MODEL_FILE":
                    _check_path(val_clean, f"[{cfg_stem}] {key}", errors, warnings)
                else:
                    # output dirs are just warnings – they'll be created at runtime
                    _check_path(val_clean, f"[{cfg_stem}] {key} (will be created)", errors, warnings, is_warning=True)

    # 3. Video + GT pairs
    for vid in videos:
        _check_path(vid["video"], "video", errors, warnings)
        gt = vid.get("gt")
        if gt:
            _check_path(gt, "gt annotation", errors, warnings)

    return errors, warnings


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run main.py for all (cfg × video) combinations")
    parser.add_argument("--config", type=str, default="Bird_HKE/experiments/run_config.json",
                        help="Path to the JSON file with run parameters")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate all paths and print commands without executing")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        run_cfg = json.load(f)

    configs = run_cfg["configs"]
    videos  = run_cfg["videos"]
    flags   = run_cfg.get("flags", [])
    filter_type = run_cfg.get("filter_type", "custom")

    total = len(configs) * len(videos)
    print(f"=== {total} runs ({len(configs)} configs × {len(videos)} videos) ===\n")

    # ── validation (always, but stop on dry-run if errors) ───────────────
    errors, warnings = validate_all(configs, videos, flags)

    if warnings:
        print("⚠  Warnings:")
        for w in warnings:
            print(w)
        print()

    if errors:
        print("✗  Errors (missing files/dirs):")
        for e in errors:
            print(e)
        print()
        if args.dry_run:
            print(f"Dry-run FAILED — fix the {len(errors)} error(s) above before running.")
            sys.exit(1)
        else:
            resp = input(f"{len(errors)} path error(s) found. Continue anyway? [y/N] ")
            if resp.strip().lower() != "y":
                sys.exit(1)
    elif args.dry_run:
        print("✓  All paths validated successfully.\n")

    # ── summary file lives next to the run_config.json ───────────────────
    summary_file = Path(args.config).parent / 'run_results_summary.csv'
    summary_rows = []

    # ── list / execute ───────────────────────────────────────────────────
    failed = []

    for i, (cfg, vid) in enumerate(itertools.product(configs, videos), 1):
        cfg_name  = Path(cfg).stem
        vid_name  = Path(vid["video"]).stem
        label     = f"[{i}/{total}] {cfg_name}  ×  {vid_name}"

        # Per-video bbox expansion (default 1.0 = no extra expansion)
        bbox_exp = vid.get("bbox_expand", 1.0)

        cmd = [
            sys.executable, "Bird_HKE/main.py",
            "--cfg", cfg,
            "--video", vid["video"],
            "--bbox-expand", str(bbox_exp),
            "--filter_type", filter_type,
            *flags,
        ]
        # Only pass --gt if ground truth is provided
        gt_path = vid.get("gt")
        if gt_path:
            cmd.insert(cmd.index("--bbox-expand"), "--gt")
            cmd.insert(cmd.index("--bbox-expand"), gt_path)

        print(f"\n{'='*70}")
        print(f"{label}")
        print(f"{'='*70}")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed.append(label)
            print(f"*** FAILED (exit {result.returncode}) ***")
            # Still collect whatever partial results were written
        row = _collect_run_results(cfg, vid, flags)
        if result.returncode != 0:
            row['flags'] = (row['flags'] + ' [FAILED]').strip()
        summary_rows.append(row)
        _write_summary(summary_file, summary_rows)
        print(f"  => summary updated: {summary_file}")

    # ── summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    if args.dry_run:
        print(f"DRY-RUN complete — {total} commands listed, all paths OK.")
    else:
        print(f"DONE  —  {total - len(failed)}/{total} succeeded")
        if failed:
            print("Failed runs:")
            for f_label in failed:
                print(f"  ✗ {f_label}")
        print(f"Summary saved to: {summary_file}")

        # ── analysis tables ──────────────────────────────────────────
        analysis_dir = Path(args.config).parent
        _write_average_table(summary_rows, analysis_dir / 'run_results_average.csv')
        _write_hypothesis_os_vs_cs(summary_rows, analysis_dir / 'hypothesis_OS_vs_CS.csv')
        _write_hypothesis_fd_hrnet(summary_rows, analysis_dir / 'hypothesis_FD_HRNet.csv')
    print(f"{'='*70}")


# ── analysis helpers ─────────────────────────────────────────────────────────

def _safe_float(val):
    """Convert to float, return None if empty or invalid."""
    if val == '' or val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _avg(values):
    """Mean of non-None values, or '' if none."""
    nums = [v for v in values if v is not None]
    if not nums:
        return ''
    return round(sum(nums) / len(nums), 4)


def _write_average_table(rows, path):
    """Write a CSV averaging metrics over all videos for each config."""
    if not rows:
        return

    metric_cols = [k for k in _SUMMARY_KEYS if k not in ('config', 'video', 'flags')]
    header = ['config'] + metric_cols

    # Group rows by config
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r['config']].append(r)

    avg_rows = []
    for cfg_name, cfg_rows in groups.items():
        avg_row = {'config': cfg_name}
        for col in metric_cols:
            vals = [_safe_float(r.get(col)) for r in cfg_rows]
            avg_row[col] = _avg(vals)
        avg_rows.append(avg_row)

    try:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(avg_rows)
        print(f"Average table saved to: {path}")
    except Exception as exc:
        print(f"Warning: could not write average table: {exc}")


def _dataset_tag(config_name):
    """Extract dataset tag (CS/OS/FD) from config stem."""
    name = config_name.upper()
    if '_CS' in name or name.endswith('CS'):
        return 'CS'
    if '_OS' in name or name.endswith('OS'):
        return 'OS'
    if '_FD' in name or name.endswith('FD'):
        return 'FD'
    return None


def _model_family(config_name):
    """Extract a readable model family name from config stem."""
    name = config_name.lower()
    if 'vhr' in name:
        return 'VHR-BirdPose'
    if 'mamba_vit' in name:
        return 'HR-MambaViT'
    if 'mamba_vision' in name:
        return 'HR-MambaVision'
    if 'mamba' in name and 'concat_gate' in name:
        return 'HR-Mamba-CG'
    if 'mamba' in name and 'sum' in name:
        return 'HR-Mamba-Sum'
    if 'mamba' in name:
        return 'HR-Mamba'
    if 'hrnet' in name:
        return 'HRNet'
    return config_name


def _write_hypothesis_os_vs_cs(rows, path):
    """H1: OS→CS improves PCK slightly but jitter improves more (initial only).

    Rows grouped by model family. For each, show OS and CS metrics side by side
    plus deltas (CS - OS).
    """
    if not rows:
        return

    from collections import defaultdict
    # group by (model_family, dataset_tag), averaging over videos
    groups = defaultdict(list)
    for r in rows:
        tag = _dataset_tag(r['config'])
        if tag not in ('CS', 'OS'):
            continue
        family = _model_family(r['config'])
        groups[(family, tag)].append(r)

    # Compute per-(family, tag) averages
    cols_of_interest = [
        'initial_PCK@0.05_mean', 'initial_PCK@0.05_visible',
        'initial_PCK@0.05_occluded', 'initial_normalized_jitter',
    ]
    avgs = {}
    for (family, tag), g_rows in groups.items():
        avgs[(family, tag)] = {}
        for col in cols_of_interest:
            vals = [_safe_float(r.get(col)) for r in g_rows]
            avgs[(family, tag)][col] = _avg(vals)

    families = sorted({f for f, _ in avgs.keys()})
    header = ['model']
    for col in cols_of_interest:
        short = col.replace('initial_', '')
        header += [f'OS_{short}', f'CS_{short}', f'delta_{short}']

    out_rows = []
    for fam in families:
        os_vals = avgs.get((fam, 'OS'), {})
        cs_vals = avgs.get((fam, 'CS'), {})
        if not os_vals or not cs_vals:
            continue
        row = {'model': fam}
        for col in cols_of_interest:
            short = col.replace('initial_', '')
            os_v = _safe_float(os_vals.get(col))
            cs_v = _safe_float(cs_vals.get(col))
            row[f'OS_{short}'] = os_v if os_v is not None else ''
            row[f'CS_{short}'] = cs_v if cs_v is not None else ''
            if os_v is not None and cs_v is not None:
                row[f'delta_{short}'] = round(cs_v - os_v, 4)
            else:
                row[f'delta_{short}'] = ''
        out_rows.append(row)

    try:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"Hypothesis OS vs CS table saved to: {path}")
    except Exception as exc:
        print(f"Warning: could not write hypothesis table: {exc}")


def _write_hypothesis_fd_hrnet(rows, path):
    """H2: In FD, HRNet vs others — visible/occluded PCK + jitter (initial only).

    Show each model's FD metrics side-by-side with HRNet-FD, plus deltas.
    """
    if not rows:
        return

    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        tag = _dataset_tag(r['config'])
        if tag != 'FD':
            continue
        family = _model_family(r['config'])
        groups[family].append(r)

    cols_of_interest = [
        'initial_PCK@0.05_mean', 'initial_PCK@0.05_visible',
        'initial_PCK@0.05_occluded', 'initial_normalized_jitter',
    ]

    avgs = {}
    for family, g_rows in groups.items():
        avgs[family] = {}
        for col in cols_of_interest:
            vals = [_safe_float(r.get(col)) for r in g_rows]
            avgs[family][col] = _avg(vals)

    hrnet_vals = avgs.get('HRNet')
    if not hrnet_vals:
        print("Warning: No HRNet FD results found for hypothesis table")
        return

    header = ['model']
    for col in cols_of_interest:
        short = col.replace('initial_', '')
        header += [short, f'HRNet_{short}', f'delta_vs_HRNet']
    # flatten: one set of delta columns per metric
    # rebuild header properly
    header = ['model']
    for col in cols_of_interest:
        short = col.replace('initial_', '')
        header += [f'{short}', f'HRNet_{short}', f'delta_{short}']

    out_rows = []
    for family in sorted(avgs.keys()):
        row = {'model': family}
        for col in cols_of_interest:
            short = col.replace('initial_', '')
            v = _safe_float(avgs[family].get(col))
            h = _safe_float(hrnet_vals.get(col))
            row[short] = v if v is not None else ''
            row[f'HRNet_{short}'] = h if h is not None else ''
            if v is not None and h is not None:
                row[f'delta_{short}'] = round(v - h, 4)
            else:
                row[f'delta_{short}'] = ''
        out_rows.append(row)

    try:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"Hypothesis FD HRNet table saved to: {path}")
    except Exception as exc:
        print(f"Warning: could not write hypothesis table: {exc}")


if __name__ == "__main__":
    main()
