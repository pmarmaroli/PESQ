from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from scipy.stats import linregress, pearsonr, spearmanr

from pesq import PesqError, pesq


@dataclass
class ScoringResult:
    pesq_score: Optional[float]
    status: str
    error: str
    ref_fs: Optional[int]
    deg_fs: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified POLQA vs PESQ regression analysis"
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(r"\\ic3-bm\Benchmarking\polqa-ml-training-data\polqa_dataset_benchmarking_team_legacy_with_nb.csv"),
        help="Input CSV path containing path_to_ref, path_to_deg, polqa_mos columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests") / "outputs",
        help="Directory where sampled CSV, scored CSV, and plot are saved",
    )
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=2000,
        help="Maximum number of random samples per POLQA MOS bin",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for stratified sampling",
    )
    parser.add_argument(
        "--target-fs",
        type=int,
        default=16000,
        choices=[16000],
        help="Target sample rate for PESQ run; wideband mode requires 16000",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip PESQ scoring and regenerate plot/metrics from an existing scored CSV",
    )
    parser.add_argument(
        "--scored-csv",
        type=Path,
        default=Path("tests") / "outputs" / "polqa_pesq_scored.csv",
        help="Path to existing scored CSV used when --plot-only is set",
    )
    return parser.parse_args()


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float64)


def _resample_if_needed(audio: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    if src_fs == dst_fs:
        return audio
    gcd = np.gcd(src_fs, dst_fs)
    up = dst_fs // gcd
    down = src_fs // gcd
    return signal.resample_poly(audio, up, down)


def _resolve_audio_path(raw_path: str, csv_path: Path) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path
    return (csv_path.parent / path).resolve()


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    required_cols = {"path_to_ref", "path_to_deg", "polqa_mos"}
    df = pd.read_csv(csv_path)
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[["path_to_ref", "path_to_deg", "polqa_mos"]].copy()
    df["polqa_mos"] = pd.to_numeric(df["polqa_mos"], errors="coerce")
    df = df.dropna(subset=["path_to_ref", "path_to_deg", "polqa_mos"])
    df = df[(df["polqa_mos"] >= 1.0) & (df["polqa_mos"] <= 5.0)]

    df["ref_path"] = df["path_to_ref"].map(lambda x: _resolve_audio_path(x, csv_path))
    df["deg_path"] = df["path_to_deg"].map(lambda x: _resolve_audio_path(x, csv_path))

    return df.reset_index(drop=True)


def stratified_sample(
    df: pd.DataFrame,
    samples_per_bin: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    bins = [1.0, 2.0, 3.0, 4.0, 5.0]
    labels = ["[1,2)", "[2,3)", "[3,4)", "[4,5]"]

    working = df.copy()
    working["mos_bin"] = pd.cut(
        working["polqa_mos"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    working.loc[working["polqa_mos"] == 5.0, "mos_bin"] = "[4,5]"

    sampled_parts: List[pd.DataFrame] = []
    summary: Dict[str, int] = {}
    for label in labels:
        part = working[working["mos_bin"] == label]
        n = min(samples_per_bin, len(part))
        summary[label] = int(n)
        if n == 0:
            continue
        sampled_parts.append(part.sample(n=n, random_state=seed, replace=False))

    if not sampled_parts:
        return working.iloc[0:0].copy(), summary

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled, summary


def score_row(ref_path: Path, deg_path: Path, target_fs: int) -> ScoringResult:
    if not ref_path.exists():
        return ScoringResult(None, "missing_ref", f"Missing ref: {ref_path}", None, None)
    if not deg_path.exists():
        return ScoringResult(None, "missing_deg", f"Missing deg: {deg_path}", None, None)

    try:
        ref_fs, ref_audio = wavfile.read(ref_path)
        deg_fs, deg_audio = wavfile.read(deg_path)

        ref_audio = _normalize_audio(ref_audio)
        deg_audio = _normalize_audio(deg_audio)

        ref_audio = _resample_if_needed(ref_audio, int(ref_fs), target_fs)
        deg_audio = _resample_if_needed(deg_audio, int(deg_fs), target_fs)

        score = pesq(
            fs=target_fs,
            ref=ref_audio,
            deg=deg_audio,
            mode="wb",
            on_error=PesqError.RETURN_VALUES,
        )

        if score is None or score < 0:
            return ScoringResult(None, "pesq_error", f"PESQ returned {score}", int(ref_fs), int(deg_fs))

        return ScoringResult(float(score), "ok", "", int(ref_fs), int(deg_fs))
    except Exception as exc:  # noqa: BLE001
        return ScoringResult(None, "exception", str(exc), None, None)


def compute_metrics(valid_df: pd.DataFrame) -> Dict[str, float]:
    x = valid_df["polqa_mos"].to_numpy(dtype=float)
    y = valid_df["pesq"].to_numpy(dtype=float)

    pearson_val, _ = pearsonr(x, y)
    spearman_val, _ = spearmanr(x, y)
    lr = linregress(x, y)

    return {
        "pearson_r": float(pearson_val),
        "spearman_rho": float(spearman_val),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "r_squared": float(lr.rvalue ** 2),
    }


def make_plot(valid_df: pd.DataFrame, metrics: Dict[str, float], output_path: Path) -> None:
    x = valid_df["polqa_mos"].to_numpy(dtype=float)
    y = valid_df["pesq"].to_numpy(dtype=float)

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, alpha=0.3, s=12, edgecolors="none")

    xs = np.linspace(x.min(), x.max(), 200)
    ys = metrics["slope"] * xs + metrics["intercept"]
    plt.plot(xs, ys, color="red", linewidth=2)

    annotation = (
        f"Pearson r = {metrics['pearson_r']:.4f}\n"
        f"Spearman rho = {metrics['spearman_rho']:.4f}\n"
        f"R^2 = {metrics['r_squared']:.4f}\n"
        f"y = {metrics['slope']:.4f}x + {metrics['intercept']:.4f}"
    )
    plt.text(0.03, 0.97, annotation, transform=plt.gca().transAxes, va="top")
    plt.xlabel("POLQA MOS")
    plt.ylabel("PESQ MOS-LQO (wideband, 16 kHz)")
    plt.title("PESQ vs POLQA (Stratified Sampling by POLQA MOS)")
    plt.xlim(1.0, 5.0)
    plt.ylim(1.0, 5.0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        if not args.scored_csv.exists():
            raise FileNotFoundError(
                f"Scored CSV not found for plot-only mode: {args.scored_csv}"
            )

        result_df = pd.read_csv(args.scored_csv)
        required_cols = {"polqa_mos", "pesq"}
        missing_cols = required_cols.difference(result_df.columns)
        if missing_cols:
            raise ValueError(
                f"Scored CSV missing required columns: {sorted(missing_cols)}"
            )

        if "status" in result_df.columns:
            valid_df = result_df[result_df["status"] == "ok"].dropna(subset=["pesq"])
        else:
            valid_df = result_df.dropna(subset=["pesq"])

        if len(valid_df) < 3:
            print("Not enough valid rows to compute regression metrics (need >= 3).")
            print(f"Valid rows: {len(valid_df)}")
            return

        metrics = compute_metrics(valid_df)
        plot_path = output_dir / "pesq_vs_polqa.png"
        make_plot(valid_df, metrics, plot_path)

        metrics_path = output_dir / "regression_metrics.txt"
        with metrics_path.open("w", encoding="utf-8") as f:
            f.write("Regression metrics\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
            f.write(f"valid_rows: {len(valid_df)}\n")
            f.write(f"total_rows: {len(result_df)}\n")

        print("Plot-only run completed.")
        print(f"Source scored CSV: {args.scored_csv}")
        print(f"Plot: {plot_path}")
        print(f"Metrics: {metrics_path}")
        print(f"Valid rows: {len(valid_df)} / {len(result_df)}")
        return

    df = load_and_validate(args.csv_path)
    sampled_df, summary = stratified_sample(df, args.samples_per_bin, args.seed)

    sampled_manifest_path = output_dir / "polqa_sampled_manifest.csv"
    sampled_df.to_csv(sampled_manifest_path, index=False)

    records = []
    for row in sampled_df.itertuples(index=False):
        result = score_row(Path(row.ref_path), Path(row.deg_path), args.target_fs)
        records.append(
            {
                "path_to_ref": str(row.path_to_ref),
                "path_to_deg": str(row.path_to_deg),
                "ref_path": str(row.ref_path),
                "deg_path": str(row.deg_path),
                "polqa_mos": float(row.polqa_mos),
                "mos_bin": str(row.mos_bin),
                "pesq": result.pesq_score,
                "status": result.status,
                "error": result.error,
                "ref_fs": result.ref_fs,
                "deg_fs": result.deg_fs,
            }
        )

    result_df = pd.DataFrame(records)
    scored_path = output_dir / "polqa_pesq_scored.csv"
    result_df.to_csv(scored_path, index=False)

    valid_df = result_df[result_df["status"] == "ok"].dropna(subset=["pesq"])
    if len(valid_df) < 3:
        print("Not enough valid rows to compute regression metrics (need >= 3).")
        print(f"Sampled rows: {len(result_df)}, valid rows: {len(valid_df)}")
        print(f"Sample manifest: {sampled_manifest_path}")
        print(f"Scored CSV: {scored_path}")
        print(f"Sampling summary: {summary}")
        return

    metrics = compute_metrics(valid_df)
    plot_path = output_dir / "pesq_vs_polqa.png"
    make_plot(valid_df, metrics, plot_path)

    metrics_path = output_dir / "regression_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("Sampling summary per MOS bin\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\nRegression metrics\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"valid_rows: {len(valid_df)}\n")
        f.write(f"total_rows: {len(result_df)}\n")

    print("Run completed.")
    print(f"Sample manifest: {sampled_manifest_path}")
    print(f"Scored CSV: {scored_path}")
    print(f"Plot: {plot_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Sampling summary: {summary}")
    print(f"Valid rows: {len(valid_df)} / {len(result_df)}")


if __name__ == "__main__":
    main()