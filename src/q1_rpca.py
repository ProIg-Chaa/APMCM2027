"""Run a single-heat robust PCA pilot on a Problem B XLSB attachment."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
from python_calamine import CalamineWorkbook
from scipy.linalg import norm, svd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


WAVELENGTH_COUNT = 2047
TIME_STEP_S = 0.5


def read_spectra(path):
    with CalamineWorkbook.from_path(path) as workbook:
        sheet = workbook.get_sheet_by_index(0)
        rows = sheet.to_python(skip_empty_area=False)

    columns = slice(1, WAVELENGTH_COUNT + 1)
    for header_index, row in enumerate(rows[:10]):
        header = row[columns]
        if len(header) == WAVELENGTH_COUNT and all(
            isinstance(value, (float, int)) for value in header
        ) and np.all(np.diff(header) > 0) and 300 < header[0] < header[-1] < 1200:
            break
    else:
        raise ValueError("Could not locate 2047 wavelength headers after the sequence column.")

    # The first column contains the 1-based sampling sequence.
    samples = [row for row in rows[header_index + 1 :] if isinstance(row[0], (float, int))]
    sequence = np.array([row[0] for row in samples])
    if not sequence.size or not np.array_equal(sequence, np.arange(1, sequence.size + 1)):
        raise ValueError("Expected consecutive, 1-based sample numbers.")
    spectra = np.array([row[columns] for row in samples], dtype=np.float32)
    if spectra.shape != (sequence.size, WAVELENGTH_COUNT) or not np.isfinite(spectra).all():
        raise ValueError("The spectral matrix has missing or non-finite values.")
    return sheet.name, np.asarray(header, dtype=np.float64), spectra


def robust_pca(observed, tolerance=1e-6, max_iterations=100):
    """Solve principal component pursuit with inexact augmented Lagrangian updates."""
    baseline = np.median(observed, axis=0)
    data = observed - baseline
    data_norm = norm(data)
    if data_norm == 0:
        return observed.copy(), np.zeros_like(observed), [(0, 0, 0.0)]

    penalty = 1 / np.sqrt(max(data.shape))
    spectral_norm = svd(data, compute_uv=False, check_finite=False)[0]
    dual = data / max(spectral_norm, np.max(np.abs(data)) / penalty)
    mu = 1.25 / spectral_norm
    max_mu = mu * 1e7
    sparse = np.zeros_like(data)
    history = []

    for iteration in range(1, max_iterations + 1):
        shifted = data + dual / mu
        left, singular, right = svd(shifted - sparse, full_matrices=False, check_finite=False)
        retained = singular > 1 / mu
        low_rank = (left[:, retained] * (singular[retained] - 1 / mu)) @ right[retained]

        remainder = shifted - low_rank
        sparse = np.sign(remainder) * np.maximum(np.abs(remainder) - penalty / mu, 0)
        residual = data - low_rank - sparse
        relative_error = float(norm(residual) / data_norm)
        rank = int(retained.sum())
        history.append((iteration, rank, relative_error))
        if iteration == 1 or iteration % 10 == 0 or relative_error <= tolerance:
            print(f"Iteration {iteration:3d}   rank {rank:4d}   relative residual {relative_error:.3e}", flush=True)
        if relative_error <= tolerance:
            break
        dual += mu * residual
        mu = min(mu * 1.5, max_mu)

    return low_rank + baseline, sparse, history


def stable_robust_pca(observed, tolerance=1e-4, max_iterations=100):
    """Fit nuclear and L1 penalties with an unstructured squared-error term using ADMM."""
    baseline = np.median(observed, axis=0)
    data = observed - baseline
    data_norm = norm(data)
    # For locally linear spectra with independent equal-variance noise, a second
    # wavelength difference has standard deviation sqrt(6) times the noise level.
    curvature = np.diff(observed, n=2, axis=1)
    noise_scale = max(1.0, float(np.median(np.abs(curvature - np.median(curvature)))) * 1.4826 / np.sqrt(6))
    sparse_penalty = 4 * noise_scale
    rank_penalty = sparse_penalty * np.sqrt(max(data.shape))
    rho = 0.25
    low_rank, sparse, low_dual, sparse_dual = (np.zeros_like(data) for _ in range(4))
    history = []

    for iteration in range(1, max_iterations + 1):
        low_target = low_rank - low_dual
        sparse_target = sparse - sparse_dual
        correction = (data - low_target - sparse_target) / (2 + rho)
        low_primal = low_target + correction
        sparse_primal = sparse_target + correction

        left, singular, right = svd(low_primal + low_dual, full_matrices=False, check_finite=False)
        retained = singular > rank_penalty / rho
        next_low = (left[:, retained] * (singular[retained] - rank_penalty / rho)) @ right[retained]
        shifted = sparse_primal + sparse_dual
        next_sparse = np.sign(shifted) * np.maximum(np.abs(shifted) - sparse_penalty / rho, 0)

        low_dual += low_primal - next_low
        sparse_dual += sparse_primal - next_sparse
        primal_residual = np.hypot(norm(low_primal - next_low), norm(sparse_primal - next_sparse))
        dual_residual = rho * np.hypot(norm(low_rank - next_low), norm(sparse - next_sparse))
        relative_error = float(max(primal_residual, dual_residual) / max(1, data_norm))
        low_rank, sparse = next_low, next_sparse
        rank = int(retained.sum())
        history.append((iteration, rank, relative_error))
        if iteration == 1 or iteration % 10 == 0 or relative_error <= tolerance:
            print(f"Iteration {iteration:3d}   rank {rank:4d}   relative ADMM residual {relative_error:.3e}", flush=True)
        if relative_error <= tolerance:
            break

    penalties = {"noise_scale": float(noise_scale), "rank_penalty": float(rank_penalty),
                 "sparse_penalty": float(sparse_penalty)}
    return low_rank + baseline, sparse, history, penalties


def find_candidates(sparse):
    """Flag large corrections relative to each wavelength's typical correction."""
    median = np.median(sparse, axis=0)
    mad = np.median(np.abs(sparse - median), axis=0)
    # A pooled floor keeps quantized channels with zero MAD from flagging tiny residuals.
    scale = np.maximum(1.4826 * mad, max(1.0, float(np.median(1.4826 * mad))))
    score = np.abs(sparse - median) / scale
    return score, score > 8.0


def screen_isolated_spikes(raw, sparse, mask):
    """Keep interior candidates with a sharp, sign-consistent local spectral deviation."""
    deviation = raw[:, 1:-1] - (raw[:, :-2] + raw[:, 2:]) / 2
    median = np.median(deviation, axis=0)
    mad = np.median(np.abs(deviation - median), axis=0)
    scale = np.maximum(1.4826 * mad, 1.0)
    spectral_spike = (np.abs(deviation - median) > 8 * scale) & (np.abs(deviation) > 0.5 * np.abs(sparse[:, 1:-1]))
    same_sign = np.sign(deviation) == np.sign(sparse[:, 1:-1])
    neighbor = (mask[:, :-2] & (np.sign(sparse[:, :-2]) == np.sign(sparse[:, 1:-1]))) | (
        mask[:, 2:] & (np.sign(sparse[:, 2:]) == np.sign(sparse[:, 1:-1]))
    )
    isolated = np.zeros_like(mask)
    isolated[:, 1:-1] = mask[:, 1:-1] & spectral_spike & same_sign & ~neighbor
    return isolated


def plot_diagnostics(path, wavelength, raw, low_rank, sparse, mask, location):
    row, column = location
    time_s = TIME_STEP_S * np.arange(raw.shape[0])
    spectral_window = slice(max(0, column - 20), min(wavelength.size, column + 21))
    time_window = slice(max(0, row - 20), min(time_s.size, row + 21))
    scale = max(1, np.quantile(np.abs(sparse), 0.997))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), layout="constrained")
    fig.suptitle("Robust PCA diagnostics", fontsize=15)
    residual_img = axes[0, 0].pcolormesh(
        wavelength, time_s, sparse, shading="nearest", cmap="RdBu_r", vmin=-scale, vmax=scale,
    )
    axes[0, 0].set(title="Sparse term S", xlabel="Wavelength (nm)", ylabel="Time (s)",
                   ylim=(time_s[-1], time_s[0]))
    fig.colorbar(residual_img, ax=axes[0, 0], label="Intensity correction")

    rows, columns = np.nonzero(mask)
    axes[0, 1].scatter(wavelength[columns], time_s[rows], s=1, alpha=0.4, linewidths=0)
    axes[0, 1].set(title="Outlier candidates", xlabel="Wavelength (nm)", ylabel="Time (s)",
                   xlim=(wavelength[0], wavelength[-1]), ylim=(time_s[-1], time_s[0]))

    axes[1, 0].plot(wavelength[spectral_window], raw[row, spectral_window], label="Observed")
    axes[1, 0].plot(wavelength[spectral_window], low_rank[row, spectral_window], label="Low-rank estimate")
    axes[1, 0].axvline(wavelength[column], color="0.55", linestyle=":")
    axes[1, 0].set(title=f"Spectrum at {time_s[row]:.1f} s", xlabel="Wavelength (nm)", ylabel="Intensity")
    axes[1, 0].legend()

    axes[1, 1].plot(time_s[time_window], raw[time_window, column], label="Observed")
    axes[1, 1].plot(time_s[time_window], low_rank[time_window, column], label="Low-rank estimate")
    axes[1, 1].axvline(time_s[row], color="0.55", linestyle=":")
    axes[1, 1].set(title=f"Time trace at {wavelength[column]:.3f} nm", xlabel="Time (s)", ylabel="Intensity")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--method", choices=("pcp", "stable"), default="pcp")
    args = parser.parse_args()

    sheet_name, wavelength, raw = read_spectra(args.workbook)
    print(f"Loaded {sheet_name}: {raw.shape}; {wavelength[0]} to {wavelength[-1]} nm", flush=True)
    started = perf_counter()
    if args.method == "stable":
        tolerance = 1e-4
        low_rank, sparse, history, penalties = stable_robust_pca(raw, tolerance)
    else:
        tolerance = 1e-6
        low_rank, sparse, history = robust_pca(raw, tolerance)
        penalties = {"sparse_to_rank_penalty_ratio": float(1 / np.sqrt(max(raw.shape)))}
    score, mask = find_candidates(sparse)
    isolated = screen_isolated_spikes(raw, sparse, mask)
    fit_seconds = perf_counter() - started

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "decomposition.npz", wavelength_nm=wavelength,
        time_s=TIME_STEP_S * np.arange(raw.shape[0]), observed=raw, low_rank=low_rank,
        sparse=sparse, candidate_mask=mask, isolated_mask=isolated,
    )

    strongest = np.argsort(score, axis=None)[-10:][::-1]
    locations = list(zip(*np.unravel_index(strongest, raw.shape)))
    iteration, rank, residual = history[-1]
    summary = {
        "workbook": args.workbook.name,
        "sheet": sheet_name,
        "method": args.method,
        "penalties": penalties,
        "samples": raw.shape[0],
        "wavelengths": raw.shape[1],
        "wavelength_range_nm": [float(wavelength[0]), float(wavelength[-1])],
        "iteration": iteration,
        "centered_low_rank": rank,
        "relative_solver_residual": residual,
        "converged": residual <= tolerance,
        "unstructured_residual_rmse": float(np.sqrt(np.mean((raw - low_rank - sparse)**2))),
        "sparse_nonzero_fraction": float(np.mean(sparse != 0)),
        "candidate_count": int(mask.sum()),
        "candidate_fraction": float(mask.mean()),
        "isolated_candidate_count": int(isolated.sum()),
        "fit_seconds": round(fit_seconds, 2),
        "strongest_points": [
            {"sample": int(row + 1), "time_s": TIME_STEP_S * int(row),
             "wavelength_nm": float(wavelength[column]), "observed": float(raw[row, column]),
             "low_rank": float(low_rank[row, column]), "sparse": float(sparse[row, column]),
             "score": float(score[row, column]), "passes_isolation_screen": bool(isolated[row, column])}
            for row, column in locations
        ],
        "iteration_history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_diagnostics(output_dir / "diagnostics.png", wavelength, raw, low_rank, sparse, mask, locations[0])
    print(f"Saved {int(mask.sum())} candidates ({mask.mean():.2%}) to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
