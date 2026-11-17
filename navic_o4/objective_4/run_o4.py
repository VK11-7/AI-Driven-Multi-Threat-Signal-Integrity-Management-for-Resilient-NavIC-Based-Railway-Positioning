"""Objective 4: corrected, transparent proxy evaluation.

Run from project root: python objective_4/run_o4.py
This script reuses saved O1/O2 outputs; it does not train models.

Important: outputs are signal/visibility proxies, not certified PVT, HPL/VPL,
Kavach safety, or measured railway throughput results.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "navic_o1" / "data" / "processed" / "all_raw_cleaned.parquet"
O1_PRED_PATH = ROOT / "navic_o1" / "results" / "o1_full_dataset" / "predictions" / "o1_all_model_predictions.parquet"
O2_PRED_PATH = ROOT / "navic_o2" / "results" / "o2_operational_test_predictions.parquet"
OUT_DIR = ROOT / "objective_4" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

O1_MODEL_NAME = "DeepAR"
THREAT_CONFIDENCE_THRESHOLD = 0.70
RISK_QUANTILE = 0.95       # calibrated on pre-test observations only
MIN_SATELLITES_FOR_VISIBILITY = 4
PROTECTION_PROXY_LIMIT_M = 10.0
BASELINE_TRAINS_PER_HOUR = 12.0  # illustrative assumption only
C_LIGHT = 299792458.0
L5_FREQUENCY_HZ = 1176.45e6
L5_WAVELENGTH_M = C_LIGHT / L5_FREQUENCY_HZ


def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_inputs():
    for p, label in [(DATA_PATH, "Processed RINEX dataset"),
                     (O1_PRED_PATH, "O1 prediction output"),
                     (O2_PRED_PATH, "O2 operational predictions")]:
        require_file(p, label)
    raw, o1, o2 = (pd.read_parquet(p) for p in [DATA_PATH, O1_PRED_PATH, O2_PRED_PATH])
    for d in (raw, o1, o2):
        d["time"] = pd.to_datetime(d["time"], errors="coerce")
        d.dropna(subset=["time", "station", "sv"], inplace=True)
    return raw, o1, o2


def calibrate_threshold(raw, o1):
    """Use only raw observations earlier than the first O1 test prediction."""
    o1d = o1[o1["model"].astype(str).str.lower() == O1_MODEL_NAME.lower()].copy()
    if o1d.empty:
        raise ValueError(f"No {O1_MODEL_NAME} rows in O1 prediction file.")
    test_start = o1d["time"].min()
    train_values = pd.to_numeric(
        raw.loc[raw["time"] < test_start, "sigma_phi_proxy"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()
    train_values = train_values[train_values >= 0]
    if train_values.empty:
        raise ValueError("Cannot calibrate risk threshold: no pre-test sigma_phi_proxy values.")
    threshold = float(train_values.quantile(RISK_QUANTILE))
    if not np.isfinite(threshold):
        raise ValueError("Calibrated sigma_phi_proxy threshold is not finite.")
    print(f"\nCalibration: {RISK_QUANTILE:.0%} quantile of pre-test sigma_phi_proxy")
    print(f"Pre-test cutoff: {test_start}")
    print(f"Calibration samples: {len(train_values):,}")
    print(f"Fixed risk threshold: {threshold:,.6g}")
    return threshold, test_start


def prepare_test(raw, o1, o2):
    required_o1 = {"model", "station", "sv", "time", "actual", "prediction"}
    required_o2 = {"station", "sv", "time", "class_name", "attack_label", "predicted_class", "confidence"}
    if not required_o1.issubset(o1.columns):
        raise ValueError(f"O1 file missing: {sorted(required_o1-set(o1.columns))}")
    if not required_o2.issubset(o2.columns):
        raise ValueError(f"O2 file missing: {sorted(required_o2-set(o2.columns))}")

    o1t = o1[o1["model"].astype(str).str.lower() == O1_MODEL_NAME.lower()].copy()
    o2g = o2[(o2["class_name"].astype(str) == "Genuine") &
             (pd.to_numeric(o2["attack_label"], errors="coerce") == 0)].copy()
    cols = ["time", "station", "sv", "C5A", "L5A", "S5A", "sigma_phi_proxy", "condition"]
    raw_cols = [c for c in cols if c in raw.columns]
    if not {"C5A", "L5A", "S5A", "sigma_phi_proxy"}.issubset(raw_cols):
        raise ValueError("Raw dataset must contain C5A, L5A, S5A and sigma_phi_proxy.")
    rr = raw[raw_cols].sort_values(["station", "sv", "time"]).drop_duplicates(
        ["station", "sv", "time"], keep="first").copy()
    rr["C5A"] = pd.to_numeric(rr["C5A"], errors="coerce")
    rr["L5A"] = pd.to_numeric(rr["L5A"], errors="coerce")
    rr["code_carrier_m"] = rr["C5A"] - rr["L5A"] * L5_WAVELENGTH_M
    rr["code_carrier_baseline"] = rr.groupby(["station", "sv"])["code_carrier_m"].transform(
        lambda s: s.rolling(10, min_periods=3).median())
    rr["signal_consistency_error_proxy_m"] = (rr["code_carrier_m"] - rr["code_carrier_baseline"]).abs()

    # Prevent duplicate joins from silently multiplying observations.
    o1t = o1t.drop_duplicates(["station", "sv", "time"], keep="last")
    o2g = o2g.drop_duplicates(["station", "sv", "time"], keep="last")
    x = o1t.merge(o2g[["station", "sv", "time", "predicted_class", "confidence"]],
                  on=["station", "sv", "time"], how="inner", validate="one_to_one")
    x = x.merge(rr, on=["station", "sv", "time"], how="left", validate="one_to_one")
    if x.empty:
        raise ValueError("No common O1 DeepAR/O2 genuine timestamps. Check split alignment.")
    for c in ["prediction", "actual", "sigma_phi_proxy", "confidence", "predicted_class"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["predicted_class"] = x["predicted_class"].fillna(0).astype(int)
    return x


def apply_policies(x, risk_threshold):
    z = x.copy()
    conf = z["confidence"].fillna(0)
    spoof = (z["predicted_class"] == 2) & (conf >= THREAT_CONFIDENCE_THRESHOLD)
    multi = (z["predicted_class"] == 1) & (conf >= THREAT_CONFIDENCE_THRESHOLD)
    current_risk = z["sigma_phi_proxy"] >= risk_threshold
    forecast_risk = z["prediction"] >= risk_threshold

    # Reactive baseline: act only on the currently observed degradation proxy.
    z["reactive_exclude"] = current_risk.fillna(False)
    # Proactive: use the 10-minute forecast for elevated proxy risk and O2 spoofing alarm.
    # Multipath is downweighted/flagged, not counted as a hard exclusion.
    z["proactive_exclude"] = (forecast_risk | spoof).fillna(False)
    z["proactive_downweight"] = multi.fillna(False)
    z["proactive_forecast_warning"] = forecast_risk.fillna(False)

    valid = pd.Series(True, index=z.index)
    for c in ["C5A", "L5A", "S5A"]:
        valid &= pd.to_numeric(z[c], errors="coerce").map(np.isfinite)
    z["signal_valid"] = valid
    z["reactive_usable"] = valid & ~z["reactive_exclude"]
    z["proactive_usable"] = valid & ~z["proactive_exclude"]
    return z


def make_epoch_table(x):
    """Station-time visibility proxy: >=4 usable satellite streams."""
    z = x.copy()
    z["reactive_usable_count"] = z["reactive_usable"].astype(int)
    z["proactive_usable_count"] = z["proactive_usable"].astype(int)
    z["valid_count"] = z["signal_valid"].astype(int)
    ep = z.groupby(["station", "time"], as_index=False).agg(
        valid_satellites=("valid_count", "sum"),
        reactive_usable_satellites=("reactive_usable_count", "sum"),
        proactive_usable_satellites=("proactive_usable_count", "sum"),
        condition=("condition", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "unknown")
        if "condition" in z.columns else ("sv", "count"),
    )
    # Only epochs with at least four valid source satellite streams are eligible.
    ep = ep[ep["valid_satellites"] >= MIN_SATELLITES_FOR_VISIBILITY].copy()
    ep["reactive_positioning_visibility_proxy"] = ep["reactive_usable_satellites"] >= MIN_SATELLITES_FOR_VISIBILITY
    ep["proactive_positioning_visibility_proxy"] = ep["proactive_usable_satellites"] >= MIN_SATELLITES_FOR_VISIBILITY
    return ep


def make_policy_summary(x, epochs):
    rows = []
    for name, usable, excluded, downweighted, ep_col in [
        ("Reactive current-observation baseline", "reactive_usable", "reactive_exclude", None,
         "reactive_positioning_visibility_proxy"),
        ("Proactive O1 forecast + O2 spoofing policy", "proactive_usable", "proactive_exclude",
         "proactive_downweight", "proactive_positioning_visibility_proxy")]:
        retained = x[x[usable] & x["signal_consistency_error_proxy_m"].notna()]
        compliant = (retained["signal_consistency_error_proxy_m"] <= PROTECTION_PROXY_LIMIT_M)
        rows.append({
            "Policy": name,
            "Usable satellite-observation rate (%)": 100 * x[usable].mean(),
            "4-satellite visibility proxy (%)": 100 * epochs[ep_col].mean() if len(epochs) else np.nan,
            "Signal-consistency proxy compliance among retained (%)": 100 * compliant.mean() if len(retained) else np.nan,
            "Retained signal rows": int(x[usable].sum()),
            "Excluded signal rows": int((x["signal_valid"] & x[excluded]).sum()),
            "Multipath downweighted rows": int(x["proactive_downweight"].sum()) if downweighted else 0,
            "Eligible station-time epochs": len(epochs),
        })
    return pd.DataFrame(rows)


def seasonal_summary(x):
    if "condition" not in x.columns:
        return pd.DataFrame()
    z = x.copy()
    z["condition"] = z["condition"].fillna("unknown").astype(str)
    rows = []
    for cond, g in z.groupby("condition"):
        rows.append({
            "Condition": cond, "Satellite observations": len(g),
            "Reactive usable signal rate (%)": 100 * g["reactive_usable"].mean(),
            "Proactive usable signal rate (%)": 100 * g["proactive_usable"].mean(),
            "Reactive mean sigma_phi_proxy": g["sigma_phi_proxy"].median(),
            "Proactive forecast median": g["prediction"].median(),
            "Reactive excluded (%)": 100 * g["reactive_exclude"].mean(),
            "Proactive excluded (%)": 100 * g["proactive_exclude"].mean(),
        })
    return pd.DataFrame(rows)


def synthetic_detection(o2):
    required = {"attack_label", "predicted_class"}
    if not required.issubset(o2.columns):
        return pd.DataFrame()
    a = o2[pd.to_numeric(o2["attack_label"], errors="coerce").isin([1, 2])].copy()
    if a.empty:
        return pd.DataFrame()
    a["attack_label"] = pd.to_numeric(a["attack_label"]).astype(int)
    a["predicted_class"] = pd.to_numeric(a["predicted_class"], errors="coerce").fillna(0).astype(int)
    names = {1: "Multipath", 2: "Spoofing"}
    out = []
    for cls, name in names.items():
        q = a[a["attack_label"] == cls]
        if len(q):
            out.append({"Threat": name, "Held-out synthetic attack samples": len(q),
                        "Correct class predictions": int((q["predicted_class"] == cls).sum()),
                        "Sample-level class recall (%)": 100 * (q["predicted_class"] == cls).mean(),
                        "Scope": "Synthetic held-out scenarios; not field validation"})
    return pd.DataFrame(out)


def save_plots(summary, seasonal):
    cols = ["Usable satellite-observation rate (%)", "4-satellite visibility proxy (%)"]
    ax = summary.set_index("Policy")[cols].plot(kind="bar", figsize=(10, 6), ylim=(0, 105))
    ax.set_ylabel("Percentage"); ax.set_title("O4 policy comparison - visibility proxies")
    ax.grid(axis="y", alpha=.25); plt.xticks(rotation=12, ha="right"); plt.tight_layout()
    plt.savefig(OUT_DIR / "o4_policy_visibility_comparison.png", dpi=200); plt.close()
    if not seasonal.empty:
        ax = seasonal.set_index("Condition")[["Reactive usable signal rate (%)", "Proactive usable signal rate (%)"]].plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel("Usable signal observations (%)"); ax.set_title("Seasonal robustness by condition")
        ax.grid(axis="y", alpha=.25); plt.xticks(rotation=15, ha="right"); plt.tight_layout()
        plt.savefig(OUT_DIR / "o4_seasonal_robustness.png", dpi=200); plt.close()


def main():
    print("=" * 72); print("OBJECTIVE 4 - CORRECTED PROXY EVALUATION"); print("=" * 72)
    raw, o1, o2 = load_inputs()
    threshold, test_start = calibrate_threshold(raw, o1)
    aligned = prepare_test(raw, o1, o2)
    evaluated = apply_policies(aligned, threshold)
    epochs = make_epoch_table(evaluated)
    summary = make_policy_summary(evaluated, epochs)
    seasonal = seasonal_summary(evaluated)
    attacks = synthetic_detection(o2)
    throughput = summary[["Policy", "4-satellite visibility proxy (%)"]].copy()
    throughput["Assumed baseline trains/hour (illustrative)"] = BASELINE_TRAINS_PER_HOUR
    throughput["Illustrative scaled trains/hour"] = BASELINE_TRAINS_PER_HOUR * throughput["4-satellite visibility proxy (%)"] / 100

    evaluated.to_parquet(OUT_DIR / "o4_aligned_test_predictions.parquet", index=False)
    epochs.to_csv(OUT_DIR / "o4_station_time_visibility.csv", index=False)
    summary.to_csv(OUT_DIR / "o4_policy_comparison.csv", index=False)
    seasonal.to_csv(OUT_DIR / "o4_seasonal_robustness_proxy.csv", index=False)
    attacks.to_csv(OUT_DIR / "o4_synthetic_attack_detection.csv", index=False)
    throughput.to_csv(OUT_DIR / "o4_illustrative_corridor_scenario.csv", index=False)
    pd.DataFrame([{"risk_quantile": RISK_QUANTILE, "calibrated_threshold": threshold,
                   "pre_test_cutoff": str(test_start), "confidence_threshold": THREAT_CONFIDENCE_THRESHOLD,
                   "protection_proxy_limit_m": PROTECTION_PROXY_LIMIT_M}]).to_csv(
                       OUT_DIR / "o4_calibration_settings.csv", index=False)
    save_plots(summary, seasonal)

    print("\nCALIBRATED THRESHOLD"); print(f"{RISK_QUANTILE:.0%} pre-test quantile = {threshold:,.6g}")
    print("\nPOLICY COMPARISON"); print(summary.round(3).to_string(index=False))
    print("\nSEASONAL ROBUSTNESS"); print(seasonal.round(3).to_string(index=False) if not seasonal.empty else "No condition column.")
    print("\nHELD-OUT SYNTHETIC ATTACK CLASSIFICATION"); print(attacks.round(3).to_string(index=False) if not attacks.empty else "No attack labels.")
    print("\nILLUSTRATIVE CORRIDOR SCALING (NOT MEASURED THROUGHPUT)"); print(throughput.round(3).to_string(index=False))
    print(f"\nSaved outputs: {OUT_DIR.resolve()}")
    print("\nInterpretation: >=4-satellite visibility is only a visibility proxy, not a PVT fix. "
          "Signal-consistency compliance is not HPL/VPL. Synthetic attack recall is not field validation. "
          "Train/hour scaling is illustrative only.")

if __name__ == "__main__":
    main()
