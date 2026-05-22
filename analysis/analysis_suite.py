"""
ANNITIA — Interpretability & Clinical-Insight Analysis Suite
=============================================================
Generates 8 SHAP / trajectory figures + results.json.

Usage (run from repository root, data files must be present):
    python analysis/analysis_suite.py

Outputs written to:  figures/
  fig1_signal_cindex.png   — single-feature C-index (rate vs static)
  fig2_leakage.png         — post-event observation leakage analysis
  fig3_shap_summary.png    — SHAP beeswarm (XGBoost-Cox)
  fig4_importance.png      — SHAP bar chart top-15 features
  fig5_trajectories.png    — FibroTest & GGT trajectory phenotypes
  fig6_oof.png             — 5-fold OOF C-index stability
  fig7_death.png           — death endpoint informative censoring
  fig8_dependence.png      — SHAP dependence of the rate signal
  results.json             — verified statistics used in the report
"""
import warnings; warnings.filterwarnings("ignore")
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress, rankdata
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import xgboost as xgb
import shap

# ── Configuration ──────────────────────────────────────────────────────────────
RS         = 42
TRAIN_PATH = "train_data.csv"
TEST_PATH  = "test_data.csv"
FIG_DIR    = "figures"
JSON_OUT   = "results.json"

np.random.seed(RS)
os.makedirs(FIG_DIR, exist_ok=True)

BIOMARKERS = ["alt","ast","bilirubin","chol","ggt","gluc_fast","plt","triglyc"]
NITS       = ["fibs_stiffness_med_BM_1","fibrotest_BM_2","aixp_aix_result_BM_3"]
BIO_VISITS = list(range(1, 22))
RESULTS    = {}

C = dict(primary="#1A5276", accent="#117A65", warn="#B9770E",
         danger="#922B21", neutral="#5D6D7E",
         event="#C0392B", noevent="#2E86C1", grid="#E5E8E8")
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# ── Helpers ────────────────────────────────────────────────────────────────────
def slope(v, t):
    m = ~(np.isnan(v) | np.isnan(t))
    if m.sum() < 2: return np.nan
    try: return linregress(t[m], v[m])[0]
    except: return np.nan

def auc_t(v, t):
    m = ~(np.isnan(v) | np.isnan(t))
    if m.sum() < 2: return np.nan
    v, t = v[m], t[m]; o = np.argsort(t)
    return float(np.trapezoid(v[o], t[o]))

def last_o(r):
    v = r[~np.isnan(r)]; return v[-1] if len(v) else np.nan

def first_o(r):
    v = r[~np.isnan(r)]; return v[0] if len(v) else np.nan

# ── Feature engineering ────────────────────────────────────────────────────────
def engineer(df):
    n = len(df); out = pd.DataFrame(index=df.index)
    for c in ["gender","T2DM","Hypertension","Dyslipidaemia","bariatric_surgery","bariatric_surgery_age"]:
        if c in df.columns: out[c] = df[c]
    acol = sorted([c for c in df.columns if c.startswith("Age_v")],
                  key=lambda c: int(c.split("_v")[1]))
    am = df[acol].values.astype(float)
    out["age_baseline"]      = np.nanmin(am, axis=1)
    out["age_last"]          = np.nanmax(am, axis=1)
    out["followup_duration"] = out["age_last"] - out["age_baseline"]
    out["n_visits"]          = (~np.isnan(am)).sum(axis=1)
    bcol = sorted([c for c in df.columns if c.startswith("BMI_v")],
                  key=lambda c: int(c.split("_v")[1]))
    if bcol:
        bm = df[bcol].values.astype(float)
        out["bmi_baseline"] = bm[:, 0]
        out["bmi_last"]     = [last_o(r) for r in bm]
        out["bmi_mean"]     = np.nanmean(bm, axis=1)
        out["bmi_max"]      = np.nanmax(bm, axis=1)
        out["bmi_delta"]    = out["bmi_last"] - out["bmi_baseline"]
        out["bmi_slope"]    = [slope(bm[i], am[i]) for i in range(n)]
    for bio in BIOMARKERS:
        cols = sorted([c for c in df.columns if c.startswith(f"{bio}_v")],
                      key=lambda c: int(c.split("_v")[1]))
        if not cols: continue
        mat = df[cols].values.astype(float); nv = mat.shape[1]; av = am[:, :nv]
        lv  = np.array([last_o(r) for r in mat]); fv = mat[:, 0]
        out[f"{bio}_first"] = fv; out[f"{bio}_last"] = lv; out[f"{bio}_delta"] = lv - fv
        out[f"{bio}_mean"]  = np.nanmean(mat, axis=1); out[f"{bio}_std"] = np.nanstd(mat, axis=1)
        out[f"{bio}_max"]   = np.nanmax(mat, axis=1);  out[f"{bio}_min"] = np.nanmin(mat, axis=1)
        out[f"{bio}_n_obs"] = (~np.isnan(mat)).sum(axis=1)
        out[f"{bio}_slope"] = [slope(mat[i], av[i]) for i in range(n)]
        out[f"{bio}_auc"]   = [auc_t(mat[i], av[i]) for i in range(n)]
        out[f"{bio}_trend_pos"] = (pd.Series(out[f"{bio}_slope"]) > 0).astype(float).values
        mid = nv // 2
        se  = [slope(mat[i, :mid], av[i, :mid]) for i in range(n)]
        sl  = [slope(mat[i, mid:], av[i, mid:]) for i in range(n)]
        out[f"{bio}_accel"] = np.array(sl) - np.array(se)
    for nit in NITS:
        cols = sorted([c for c in df.columns if c.startswith(nit)],
                      key=lambda c: int(c.split("_v")[1]))
        if not cols: continue
        mat = df[cols].values.astype(float); nv = mat.shape[1]; av = am[:, :nv]
        lv  = np.array([last_o(r) for r in mat]); fv = mat[:, 0]; sh = nit.split("_")[0]
        out[f"{sh}_first"]        = fv;  out[f"{sh}_last"]  = lv;  out[f"{sh}_delta"] = lv - fv
        out[f"{sh}_mean"]         = np.nanmean(mat, axis=1)
        out[f"{sh}_std"]          = np.nanstd(mat, axis=1)
        out[f"{sh}_max"]          = np.nanmax(mat, axis=1)
        out[f"{sh}_n_obs"]        = (~np.isnan(mat)).sum(axis=1)
        out[f"{sh}_slope"]        = [slope(mat[i], av[i]) for i in range(n)]
        out[f"{sh}_auc"]          = [auc_t(mat[i], av[i]) for i in range(n)]
        out[f"{sh}_trend_pos"]    = (pd.Series(out[f"{sh}_slope"]) > 0).astype(float).values
        out[f"{sh}_pct_change"]   = (lv - fv) / (np.abs(fv) + 1e-8)
        out[f"{sh}_recent_slope"] = [slope(mat[i, -3:], av[i, -3:]) for i in range(n)]
        if "fibs" in nit:
            mx = np.nanmax(mat, axis=1)
            out["fibs_f2_ever"]    = (mx > 7.1).astype(float)
            out["fibs_f3_ever"]    = (mx > 9.6).astype(float)
            out["fibs_f4_ever"]    = (mx > 13.6).astype(float)
            out["fibs_f4_last"]    = (lv > 13.6).astype(float)
            out["fibs_f3_last"]    = (lv > 9.6).astype(float)
            out["fibs_prog_f2_f4"] = ((mat[:, 0] <= 9.6) & (lv > 13.6)).astype(float)
        if "fibrotest" in nit:
            mx = np.nanmax(mat, axis=1)
            out["ft_f2_ever"] = (mx > 0.48).astype(float)
            out["ft_f3_ever"] = (mx > 0.59).astype(float)
            out["ft_f4_ever"] = (mx > 0.75).astype(float)
            out["ft_f4_last"] = (lv > 0.75).astype(float)
    age_l = out["age_last"].values
    ast_l = out.get("ast_last", pd.Series(np.nan, index=out.index)).values
    alt_l = out.get("alt_last", pd.Series(np.nan, index=out.index)).values
    plt_l = out.get("plt_last", pd.Series(np.nan, index=out.index)).values
    with np.errstate(divide="ignore", invalid="ignore"):
        fib4 = (age_l * ast_l) / (plt_l * np.sqrt(np.maximum(alt_l, 1e-8)))
        aar  = ast_l / np.maximum(alt_l, 1e-8)
    out["fib4_last"] = np.where(np.isfinite(fib4), fib4, np.nan)
    out["aar_last"]  = np.where(np.isfinite(aar),  aar,  np.nan)
    fibs_max = out.get("fibs_max", pd.Series(np.nan, index=out.index)).values
    ft_max   = out.get("fibrotest_max", pd.Series(np.nan, index=out.index)).values
    out["fibs_x_ft"] = fibs_max * ft_max
    inv_fu = 1 / np.maximum(out["followup_duration"].values, 0.5)
    out["ft_max_x_inv_fu"]  = ft_max * inv_fu
    out["ggt_max_x_inv_fu"] = out.get("ggt_max", pd.Series(np.nan, index=out.index)).values * inv_fu
    return out.replace([np.inf, -np.inf], np.nan)

def targets(df, outcome="hepatic"):
    df = df.copy()
    acol = [c for c in df.columns if c.startswith("Age_v")]
    df["last_observed_age"] = df[acol].max(axis=1)
    if outcome == "hepatic":
        ec, ao, en = "evenements_hepatiques_majeurs", "evenements_hepatiques_age_occur", "Hepatic_event"
        iev = df[ec] == 1; mask = ~(iev & df[ao].isna())
    else:
        ec, ao, en = "death", "death_age_occur", "Death"
        iev = df[ec] == 1; mask = df[ec].notna() & ~(iev & df[ao].isna())
    dv = df[mask].copy().reset_index(drop=True); ie = dv[ec] == 1
    t  = np.maximum(np.where(ie, dv[ao] - dv["Age_v1"],
                              dv["last_observed_age"] - dv["Age_v1"]).astype(float), 0.001)
    return dv, mask, Surv.from_arrays(event=ie.astype(bool).values, time=t,
                                       name_event=en, name_time="T")

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"  Train {train.shape}  Test {test.shape}")

print("Engineering features...")
Xtr = engineer(train); Xte = engineer(test)
_, mh, yh = targets(train, "hepatic")
_, md, yd = targets(train, "death")
Xh     = Xtr.loc[train.index[mh]].reset_index(drop=True)
common = [c for c in Xh.columns if c in Xte.columns]
keep   = [c for c in common if Xh[c].isna().mean() <= 0.60]
Xh     = Xh[keep]
RESULTS["n_features"]     = len(keep)
RESULTS["n_hep_events"]   = int(yh["Hepatic_event"].sum())
RESULTS["n_death_events"] = int(yd["Death"].sum())
print(f"  {len(keep)} features, {RESULTS['n_hep_events']} hepatic events")

# ── FIG 1 ──────────────────────────────────────────────────────────────────────
print("Fig 1: signal C-index...")
ev = yh["Hepatic_event"].astype(bool); T = yh["T"]
fibs   = Xh["fibs_max"].values; ftm = Xh["fibrotest_max"].values
fu     = Xh["followup_duration"].values; inv_fu = 1 / np.maximum(fu, 0.5)

def ci(s):
    s = np.where(np.isnan(s), np.nanmedian(s), s)
    return concordance_index_censored(np.asarray(ev), np.asarray(T), s)[0]

sig = {"FibroScan max":           ci(fibs),
       "FibroTest max":           ci(ftm),
       "FibroTest × 1/follow-up": ci(ftm * inv_fu),
       "GGT × 1/follow-up":       ci(Xh["ggt_max_x_inv_fu"].values),
       "1/follow-up alone":       ci(inv_fu),
       "FIB-4 last":              ci(Xh["fib4_last"].values)}
RESULTS["signal_cindex"] = {k: round(v, 4) for k, v in sig.items()}
order  = np.argsort(list(sig.values()))
names  = [list(sig.keys())[i]   for i in order]
vals   = [list(sig.values())[i] for i in order]
colors = [C["danger"] if v < 0.5 else (C["accent"] if v > 0.7 else C["primary"]) for v in vals]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(names, vals, color=colors, edgecolor="white")
ax.axvline(0.5, ls="--", color=C["neutral"], lw=1.2, label="Random (C=0.5)")
for i, v in enumerate(vals):
    ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9.5, fontweight="bold")
ax.set_xlim(0.4, 0.85); ax.set_xlabel("Concordance Index (hepatic, training)")
ax.set_title("Single-feature discrimination — rate signals rival static fibrosis")
ax.legend(loc="lower right", frameon=False)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig1_signal_cindex.png", bbox_inches="tight"); plt.close()

# ── FIG 2 ──────────────────────────────────────────────────────────────────────
print("Fig 2: post-event leakage...")
acol = sorted([c for c in train.columns if c.startswith("Age_v")])
am   = train[acol].values.astype(float)
last = np.nanmax(am, axis=1); base = train["Age_v1"].values
eva  = train["evenements_hepatiques_age_occur"].values
mlk  = ((train["evenements_hepatiques_majeurs"] == 1)
        & train["evenements_hepatiques_age_occur"].notna()).values
fr = []; pre = []; post = []
for i in range(len(train)):
    if mlk[i] and last[i] > eva[i] and last[i] - base[i] > 0:
        fr.append((last[i] - eva[i]) / (last[i] - base[i]))
        pre.append(eva[i] - base[i]); post.append(last[i] - eva[i])
RESULTS["leak_n_events"]  = int(mlk.sum())
RESULTS["leak_n_post"]    = len(fr)
RESULTS["leak_mean_frac"] = round(float(np.mean(fr)), 3)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.hist(np.array(fr) * 100, bins=12, color=C["warn"], edgecolor="white")
a1.axvline(np.mean(fr) * 100, color=C["danger"], lw=2, label=f"Mean {np.mean(fr)*100:.0f}%")
a1.set_xlabel("% of follow-up recorded AFTER the event")
a1.set_ylabel("Number of event patients"); a1.legend(frameon=False)
a1.set_title(f"Post-event leakage: {len(fr)}/{int(mlk.sum())} events affected")
idx = np.argsort(pre)[::-1][:18]; y = np.arange(len(idx))
a2.barh(y, [pre[i] for i in idx],  color=C["noevent"], label="Pre-event (valid)")
a2.barh(y, [post[i] for i in idx], left=[pre[i] for i in idx],
        color=C["danger"], label="Post-event (leakage)")
a2.set_yticks([]); a2.set_xlabel("Follow-up duration (years)")
a2.set_title("Per-patient follow-up composition"); a2.legend(frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig2_leakage.png", bbox_inches="tight"); plt.close()

# ── XGBoost-Cox + SHAP ────────────────────────────────────────────────────────
print("Training XGBoost-Cox for SHAP...")
imp   = SimpleImputer(strategy="median")
Xh_i  = pd.DataFrame(imp.fit_transform(Xh), columns=keep)
lab   = np.where(yh["Hepatic_event"], yh["T"], -yh["T"])
dtr   = xgb.DMatrix(Xh_i.values, label=lab, feature_names=keep)
params = dict(objective="survival:cox", eval_metric="cox-nloglik",
              eta=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=8, tree_method="hist", seed=RS)
xgb_model = xgb.train(params, dtr, num_boost_round=200)
expl      = shap.TreeExplainer(xgb_model)
sv        = expl.shap_values(Xh_i.values, check_additivity=False)
mean_abs  = np.abs(sv).mean(0)
imp_df    = (pd.DataFrame({"feature": keep, "importance": mean_abs})
               .sort_values("importance", ascending=False))
RESULTS["top_features"] = imp_df.head(15).to_dict("records")

# ── FIG 3 ──────────────────────────────────────────────────────────────────────
print("Fig 3: SHAP beeswarm...")
top = imp_df.head(18)["feature"].tolist(); ti = [keep.index(f) for f in top]
plt.figure(figsize=(8.5, 7))
shap.summary_plot(sv[:, ti], Xh_i.iloc[:, ti], feature_names=top,
                  show=False, plot_size=None, color_bar=True)
plt.title("SHAP summary — hepatic risk model (XGBoost-Cox)", fontweight="bold", fontsize=13)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig3_shap_summary.png", bbox_inches="tight"); plt.close()

# ── FIG 4 ──────────────────────────────────────────────────────────────────────
print("Fig 4: SHAP importance bar...")
fig, ax = plt.subplots(figsize=(8, 6))
t15 = imp_df.head(15).iloc[::-1]
ax.barh(t15["feature"], t15["importance"], color=C["primary"], edgecolor="white")
ax.set_xlabel("Mean |SHAP value|"); ax.set_title("Top 15 hepatic risk drivers")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig4_importance.png", bbox_inches="tight"); plt.close()

# ── FIG 5 ──────────────────────────────────────────────────────────────────────
print("Fig 5: trajectory phenotypes...")
fc   = sorted([c for c in train.columns if c.startswith("fibrotest_BM_2")],
              key=lambda c: int(c.split("_v")[1]))
ggtc = sorted([c for c in train.columns if c.startswith("ggt_v")],
              key=lambda c: int(c.split("_v")[1]))
ftmat  = train[fc].values.astype(float)
ggtmat = train[ggtc].values.astype(float)
ev_all = (train["evenements_hepatiques_majeurs"] == 1).values
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for i in np.where(ev_all)[0][:40]:
    v = ftmat[i]; x = am[i, :len(v)]; m = ~np.isnan(v)
    if m.sum() > 1: a1.plot(x[m] - base[i], v[m], color=C["event"], alpha=0.32, lw=1)
for i in np.where(~ev_all)[0][:80]:
    v = ftmat[i]; x = am[i, :len(v)]; m = ~np.isnan(v)
    if m.sum() > 1: a1.plot(x[m] - base[i], v[m], color=C["noevent"], alpha=0.12, lw=0.8)
a1.plot([], [], color=C["event"], label="Event")
a1.plot([], [], color=C["noevent"], label="No event")
a1.set_xlabel("Years from baseline"); a1.set_ylabel("FibroTest")
a1.set_title("FibroTest trajectories"); a1.legend(frameon=False); a1.set_xlim(0, 12)
for i in np.where(ev_all)[0][:40]:
    v = ggtmat[i]; x = am[i, :len(v)]; m = ~np.isnan(v)
    if m.sum() > 1: a2.plot(x[m] - base[i], v[m], color=C["event"], alpha=0.32, lw=1)
for i in np.where(~ev_all)[0][:80]:
    v = ggtmat[i]; x = am[i, :len(v)]; m = ~np.isnan(v)
    if m.sum() > 1: a2.plot(x[m] - base[i], v[m], color=C["noevent"], alpha=0.12, lw=0.8)
a2.set_xlabel("Years from baseline"); a2.set_ylabel("GGT (U/L)")
a2.set_title("GGT trajectories"); a2.set_xlim(0, 12); a2.set_ylim(0, 800)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig5_trajectories.png", bbox_inches="tight"); plt.close()

# ── FIG 6 ──────────────────────────────────────────────────────────────────────
print("Fig 6: OOF stability...")
kf = StratifiedKFold(5, shuffle=True, random_state=RS)
evb = yh["Hepatic_event"].astype(int); fold_ci = []
for tri, vai in kf.split(Xh_i, evb):
    g = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05,
                                         max_depth=3, min_samples_leaf=10,
                                         subsample=0.8, random_state=RS)
    g.fit(Xh_i.iloc[tri].values, yh[tri])
    p = g.predict(Xh_i.iloc[vai].values)
    fold_ci.append(concordance_index_censored(yh[vai]["Hepatic_event"], yh[vai]["T"], p)[0])
RESULTS["hep_oof_folds"] = [round(x, 4) for x in fold_ci]
RESULTS["hep_oof_mean"]  = round(float(np.mean(fold_ci)), 4)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, 6), fold_ci, color=C["primary"], edgecolor="white")
ax.axhline(np.mean(fold_ci), ls="--", color=C["danger"], label=f"Mean {np.mean(fold_ci):.3f}")
ax.set_xlabel("Fold"); ax.set_ylabel("C-index"); ax.set_ylim(0, 1)
ax.set_title("Hepatic model — 5-fold OOF stability"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig6_oof.png", bbox_inches="tight"); plt.close()

# ── FIG 7 ──────────────────────────────────────────────────────────────────────
print("Fig 7: death censoring...")
dage   = train["death_age_occur"].values; dmask = (train["death"] == 1).values
fu_all = last - base; td = np.where(dmask, dage - base, fu_all); td = np.maximum(td, 0.001)
inv_fu_all = 1 / np.maximum(fu_all, 0.5)
c_death = concordance_index_censored(dmask, td, inv_fu_all)[0]
RESULTS["death_invfu_cindex"] = round(float(c_death), 4)
RESULTS["death_censor_rate"]  = round(float((~dmask).mean()), 3)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(fu_all[~dmask], bins=25, alpha=0.6, color=C["noevent"],
        label=f"Censored ({(~dmask).sum()})", density=True)
ax.hist(td[dmask], bins=15, alpha=0.7, color=C["event"],
        label=f"Death ({dmask.sum()})", density=True)
ax.set_xlabel("Time (years)"); ax.set_ylabel("Density")
ax.set_title(f"Death: follow-up separates events (1/fu C={c_death:.3f})")
ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig7_death.png", bbox_inches="tight"); plt.close()

# ── FIG 8 ──────────────────────────────────────────────────────────────────────
print("Fig 8: SHAP dependence...")
if "ft_max_x_inv_fu" in keep:
    fidx = keep.index("ft_max_x_inv_fu")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xv = Xh_i["ft_max_x_inv_fu"].values; yv = sv[:, fidx]
    sc = ax.scatter(xv, yv, c=Xh_i["followup_duration"].values,
                    cmap="viridis_r", s=22, alpha=0.7)
    ax.axhline(0, color=C["neutral"], lw=0.8)
    ax.set_xlabel("FibroTest × (1/follow-up)"); ax.set_ylabel("SHAP value (risk contribution)")
    ax.set_title("Rate signal drives risk — shorter follow-up amplifies it")
    plt.colorbar(sc, label="Follow-up (yr)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig8_dependence.png", bbox_inches="tight"); plt.close()

# ── Save results ───────────────────────────────────────────────────────────────
with open(JSON_OUT, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\nDone. Figures → {FIG_DIR}/   Statistics → {JSON_OUT}")
for k in ["n_features","n_hep_events","hep_oof_mean","death_invfu_cindex","leak_mean_frac"]:
    print(f"  {k}: {RESULTS.get(k)}")
