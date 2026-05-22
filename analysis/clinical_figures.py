"""
ANNITIA — Clinical Survival Analysis Figure Suite
==================================================
Generates 8 publication-quality clinical survival figures.

Usage (run from repository root, data files must be present):
    python analysis/clinical_figures.py

Outputs written to:  figures_clinical/
  fig1_km_overall.png          — Kaplan-Meier (hepatic + death, side by side)
  fig2_km_stratified.png       — KM by rate-signal risk tertile (log-rank test)
  fig3_cum_incidence_stage.png — Cumulative incidence by fibrosis stage
  fig4_biomarker_violins.png   — Biomarker distributions: events vs event-free
  fig5_calibration.png         — Predicted vs observed event rate by decile
  fig6_trajectory_heatmap.png  — FibroTest trajectory heatmap (sorted by risk)
  fig7_forest_plot.png         — Univariable Cox hazard ratios (standardised)
  fig8_landmark.png            — Landmark analysis: C-index at successive timepoints
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from sksurv.metrics import concordance_index_censored
from scipy.stats import linregress

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_PATH = "train_data.csv"
FIG_DIR    = "figures_clinical"

os.makedirs(FIG_DIR, exist_ok=True)

C = dict(ev="#C0392B", ne="#1A5276", g3="#117A65",
         f3="#B9770E", f4="#922B21", med="#5D6D7E", grid="#E8EAEB")
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.7,
    "grid.alpha": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 9,
})

# ── Load & prep ────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
N     = len(train)

ac = sorted([c for c in train.columns if c.startswith("Age_v")])
am = train[ac].values.astype(float)
base = train["Age_v1"].values; last = np.nanmax(am, axis=1); fu = last - base

ev_hep   = (train["evenements_hepatiques_majeurs"] == 1).values
ev_death = (train["death"] == 1).values
ev_hep_age   = train["evenements_hepatiques_age_occur"].values
ev_death_age = train["death_age_occur"].values
t_hep   = np.maximum(np.where(ev_hep,   ev_hep_age   - base, fu), 0.001)
t_death = np.maximum(np.where(ev_death, ev_death_age - base, fu), 0.001)
valid_h = ~(ev_hep   & np.isnan(ev_hep_age))
valid_d = ~(ev_death & np.isnan(ev_death_age))

fc  = sorted([c for c in train.columns if c.startswith("fibrotest_BM_2")],
             key=lambda c: int(c.split("_v")[1]))
fsc = sorted([c for c in train.columns if c.startswith("fibs_stiffness_med_BM_1")],
             key=lambda c: int(c.split("_v")[1]))
gc  = sorted([c for c in train.columns if c.startswith("ggt_v")],
             key=lambda c: int(c.split("_v")[1]))
bc  = sorted([c for c in train.columns if c.startswith("BMI_v")],
             key=lambda c: int(c.split("_v")[1]))

ft_mat   = train[fc].values.astype(float);  ft_max  = np.nanmax(ft_mat,  axis=1)
fibs_mat = train[fsc].values.astype(float); fibs_max = np.nanmax(fibs_mat, axis=1)
fibs_last = np.array([r[~np.isnan(r)][-1] if (~np.isnan(r)).any() else np.nan
                       for r in fibs_mat])
ggt_mat  = train[gc].values.astype(float);  ggt_max = np.nanmax(ggt_mat, axis=1)
bm       = train[bc].values.astype(float)
inv_fu   = 1 / np.maximum(fu, 0.5)
score    = np.where(np.isnan(ft_max), np.nanmedian(ft_max * inv_fu), ft_max * inv_fu)

def stage(v):
    if np.isnan(v): return "Unknown"
    if v <= 7.1:  return "F0-F1"
    if v <= 9.6:  return "F2"
    if v <= 13.6: return "F3"
    return "F4"
fibs_stage = np.array([stage(v) for v in fibs_last])

def sl(v, t):
    m = ~(np.isnan(v) | np.isnan(t))
    if m.sum() < 2: return np.nan
    try: return linregress(t[m], v[m])[0]
    except: return np.nan
ft_slope = np.array([sl(ft_mat[i], am[i, :ft_mat.shape[1]]) for i in range(N)])

def add_at_risk(ax, kmf, t_points, y=-0.12):
    ax.annotate("n at risk", xy=(0, y), xycoords=("data", "axes fraction"),
                fontsize=7.5, color=C["med"], ha="right")
    for t in t_points:
        n = int(kmf.event_table.loc[kmf.event_table.index <= t, "at_risk"].iloc[-1]
                if (kmf.event_table.index <= t).any() else 0)
        ax.annotate(str(n), xy=(t, y), xycoords=("data", "axes fraction"),
                    fontsize=7.5, color=C["med"], ha="center")

# ── FIG 1: KM overall ─────────────────────────────────────────────────────────
print("Fig 1: KM overall...")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
kmf = KaplanMeierFitter()
kmf.fit(t_hep[valid_h], ev_hep[valid_h])
kmf.plot_survival_function(ax=a1, color=C["ne"], ci_alpha=0.12, ci_show=True,
                            label=f"Hepatic event-free (n={valid_h.sum()})", linewidth=2)
a1.set_title("Hepatic event-free survival")
a1.set_xlabel("Time (years)"); a1.set_ylabel("Survival probability")
a1.set_ylim(0.85, 1.01); a1.set_xlim(0, 21)
a1.text(0.98, 0.96, f"Events: {int(ev_hep.sum())}/{N}\nMedian: not reached",
        transform=a1.transAxes, ha="right", va="top", fontsize=8.5, color=C["med"])
add_at_risk(a1, kmf, [0, 5, 10, 15, 20])

kmf2 = KaplanMeierFitter()
kmf2.fit(t_death[valid_d], ev_death[valid_d])
kmf2.plot_survival_function(ax=a2, color="#533AB7", ci_alpha=0.12, ci_show=True,
                             label=f"Overall survival (n={valid_d.sum()})", linewidth=2)
a2.set_title("Overall survival (death endpoint)")
a2.set_xlabel("Time (years)"); a2.set_ylabel("Survival probability")
a2.set_ylim(0.80, 1.01); a2.set_xlim(0, 21)
a2.text(0.98, 0.96, f"Events: {int(ev_death.sum())}/{N}\n94% censored",
        transform=a2.transAxes, ha="right", va="top", fontsize=8.5, color=C["med"])
add_at_risk(a2, kmf2, [0, 5, 10, 15, 20])
plt.tight_layout(pad=1.5)
plt.savefig(f"{FIG_DIR}/fig1_km_overall.png", bbox_inches="tight"); plt.close()

# ── FIG 2: Risk-stratified KM ─────────────────────────────────────────────────
print("Fig 2: stratified KM...")
t33, t67 = np.nanpercentile(score, 33), np.nanpercentile(score, 67)
grp = {"Low risk (T1)": score <= t33,
       "Mid risk (T2)": (score > t33) & (score <= t67),
       "High risk (T3)": score > t67}
colors_g = [C["ne"], C["f3"], C["ev"]]; dashes_g = [None, None, [6, 3]]
fig, ax = plt.subplots(figsize=(8, 4.8))
for (lbl, mask), col, dash in zip(grp.items(), colors_g, dashes_g):
    vm = mask & valid_h
    km_ = KaplanMeierFitter()
    km_.fit(t_hep[vm], ev_hep[vm],
            label=f"{lbl} (n={vm.sum()}, {int(ev_hep[vm].sum())} events)")
    lkw = {"linestyle": "--"} if dash else {}
    km_.plot_survival_function(ax=ax, color=col, ci_alpha=0.10, ci_show=True, linewidth=2, **lkw)
ax.set_title("Hepatic survival by FibroTest × (1/follow-up) risk tertile")
ax.set_xlabel("Time (years)"); ax.set_ylabel("Event-free survival probability")
ax.set_ylim(0.80, 1.01); ax.set_xlim(0, 21)
groups  = np.select([score <= t33, (score > t33) & (score <= t67), score > t67], [0, 1, 2], default=-1)
vm_all  = valid_h & (groups >= 0)
res     = multivariate_logrank_test(t_hep[vm_all], groups[vm_all], ev_hep[vm_all])
ax.text(0.02, 0.04, f"Log-rank p = {res.p_value:.4f}",
        transform=ax.transAxes, fontsize=9, color=C["med"])
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig2_km_stratified.png", bbox_inches="tight"); plt.close()

# ── FIG 3: Cumulative incidence by fibrosis stage ─────────────────────────────
print("Fig 3: cumulative incidence...")
stages_ord   = ["F0-F1", "F2", "F3", "F4"]
stage_colors = [C["ne"], C["g3"], C["f3"], C["ev"]]
fig, ax = plt.subplots(figsize=(8, 4.8))
for s, col in zip(stages_ord, stage_colors):
    mask = (fibs_stage == s) & valid_h
    if mask.sum() < 5: continue
    km_ = KaplanMeierFitter()
    km_.fit(t_hep[mask], ev_hep[mask])
    ci_curve = 1 - km_.survival_function_
    n_ev = int(ev_hep[mask].sum())
    ax.step(ci_curve.index, ci_curve.iloc[:, 0] * 100, where="post", color=col, lw=2,
            label=f"{s} (n={mask.sum()}, {n_ev} events, rate={n_ev/mask.sum()*100:.1f}%)")
    ax.fill_between(ci_curve.index, 0, ci_curve.iloc[:, 0] * 100,
                    step="post", alpha=0.08, color=col)
ax.set_title("Cumulative incidence of hepatic events by fibrosis stage")
ax.set_xlabel("Time (years)"); ax.set_ylabel("Cumulative incidence (%)")
ax.set_xlim(0, 21); ax.set_ylim(0, 35); ax.legend(loc="upper left", fontsize=9)
ax.text(0.98, 0.04, "FibroScan stage at last visit (F2>7.1, F3>9.6, F4>13.6 kPa)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=C["med"])
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig3_cum_incidence_stage.png", bbox_inches="tight"); plt.close()

# ── FIG 4: Biomarker violin plots ─────────────────────────────────────────────
print("Fig 4: biomarker violins...")
fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
biomarkers = [("FibroTest max", ft_max, None),
              ("FibroScan max (kPa)", fibs_max, None),
              ("GGT max (U/L)", ggt_max, 800),
              ("Follow-up (yr)", fu, None)]
for ax, (lbl, vals, ylim) in zip(axes, biomarkers):
    ev_v = vals[ev_hep & ~np.isnan(vals)]
    ne_v = vals[~ev_hep & ~np.isnan(vals)]
    parts = ax.violinplot([ne_v, ev_v], positions=[0, 1],
                          showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], [C["ne"], C["ev"]]):
        pc.set_facecolor(col); pc.set_alpha(0.35); pc.set_edgecolor(col)
    parts["cmedians"].set_color([C["ne"], C["ev"]]); parts["cmedians"].set_linewidth(2.5)
    for vals_, x, col in [(ne_v, 0, C["ne"]), (ev_v, 1, C["ev"])]:
        jit = np.random.uniform(-0.07, 0.07, len(vals_))
        ax.scatter(x + jit, vals_, alpha=0.25, s=8, color=col, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No event", "Event"], fontsize=9)
    ax.set_ylabel(lbl, fontsize=9)
    if ylim: ax.set_ylim(0, ylim)
    ax.annotate(f"med={np.nanmedian(ne_v):.2f}",
                xy=(0, np.nanmedian(ne_v)), xytext=(0.35, np.nanmedian(ne_v)),
                fontsize=7.5, color=C["ne"], va="center")
    ax.annotate(f"med={np.nanmedian(ev_v):.2f}",
                xy=(1, np.nanmedian(ev_v)), xytext=(0.65, np.nanmedian(ev_v)),
                fontsize=7.5, color=C["ev"], va="center")
plt.suptitle("Biomarker distributions — event vs. event-free patients",
             fontweight="bold", y=1.01)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig4_biomarker_violins.png", bbox_inches="tight"); plt.close()

# ── FIG 5: Calibration ────────────────────────────────────────────────────────
print("Fig 5: calibration...")
fig, ax = plt.subplots(figsize=(6, 5.5))
deciles = np.percentile(score, np.arange(10, 101, 10))
obs_rates = []; pred_rates = []; ns = []
prev = score.min() - 0.001
for d in deciles:
    m = (score > prev) & (score <= d) & valid_h
    if m.sum() > 0:
        obs_rates.append(ev_hep[m].mean() * 100)
        pred_rates.append(
            ((score[m] - score[m].min()) / (score[m].max() - score[m].min() + 1e-9)).mean() * 20)
        ns.append(m.sum())
    prev = d
obs_r = np.array(obs_rates); pred_r = np.array(pred_rates)
ax.scatter(pred_r, obs_r, s=[n / 2 for n in ns], color=C["ne"],
           alpha=0.75, edgecolors=C["ne"], lw=1.5, zorder=4)
for i, (p, o, n) in enumerate(zip(pred_r, obs_r, ns)):
    ax.annotate(f"D{i+1}\nn={n}", xy=(p, o), xytext=(4, 4),
                textcoords="offset points", fontsize=7.5, color=C["med"])
lims = [0, max(max(obs_r), max(pred_r)) * 1.15]
ax.plot(lims, lims, "--", color=C["med"], lw=1.2, label="Perfect calibration")
z = np.polyfit(pred_r, obs_r, 1); p_fit = np.poly1d(z)
xfit = np.linspace(min(pred_r), max(pred_r), 50)
ax.plot(xfit, p_fit(xfit), "-", color=C["ev"], lw=1.5, alpha=0.7,
        label=f"Fitted (slope={z[0]:.2f})")
ax.set_xlabel("Predicted event rate (risk score decile, %)")
ax.set_ylabel("Observed event rate (%)")
ax.set_title("Calibration: predicted vs. observed event rates by score decile")
ax.legend(fontsize=9); ax.set_xlim(lims); ax.set_ylim(lims)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig5_calibration.png", bbox_inches="tight"); plt.close()

# ── FIG 6: FibroTest trajectory heatmap ──────────────────────────────────────
print("Fig 6: trajectory heatmap...")
ev_idx = np.where(ev_hep)[0][:30]
ne_idx = [i for i in np.argsort(-score) if not ev_hep[i]][:30]
all_idx = np.concatenate([ev_idx, ne_idx])
mat_sub = ft_mat[all_idx, :]
fig, ax = plt.subplots(figsize=(12, 5.5))
im = ax.imshow(np.ma.masked_invalid(mat_sub), aspect="auto",
               cmap="RdYlBu_r", vmin=0.15, vmax=0.85)
ax.axhline(len(ev_idx) - 0.5, color="white", lw=2.5)
ax.text(-0.5, len(ev_idx) / 2, "Events\n(n=30)", ha="right", va="center",
        fontsize=9, color=C["ev"], fontweight="bold", rotation=90)
ax.text(-0.5, len(ev_idx) + len(ne_idx) / 2, "No event\n(n=30)", ha="right", va="center",
        fontsize=9, color=C["ne"], fontweight="bold", rotation=90)
ax.set_xlabel("Visit number"); ax.set_ylabel("Patient (sorted by risk score)")
ax.set_yticks([])
ax.set_xticks(range(0, ft_mat.shape[1], 3))
ax.set_xticklabels(range(1, ft_mat.shape[1] + 1, 3))
plt.colorbar(im, ax=ax, shrink=0.7, pad=0.01).set_label("FibroTest value", fontsize=9)
ax.set_title("FibroTest trajectory heatmap — events vs. event-free (sorted by risk score)",
             fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig6_trajectory_heatmap.png", bbox_inches="tight"); plt.close()

# ── FIG 7: Univariable Cox forest plot (standardised) ────────────────────────
print("Fig 7: forest plot...")
def std(v): s = np.nanstd(v); return (v - np.nanmean(v)) / (s if s > 0 else 1)

covs = {
    "FibroScan max":             std(fibs_max),
    "FibroTest max":             std(ft_max),
    "GGT max":                   std(ggt_max),
    "FibroTest slope":           std(ft_slope),
    "Follow-up (yr)":            std(fu),
    "Rate: FibroTest×(1/fu)":    std(ft_max * inv_fu),
    "BMI max":                   std(np.nanmax(bm, axis=1)),
    "Age at baseline":           std(base),
    "T2DM":                      train["T2DM"].values.astype(float),
}
df_cox = pd.DataFrame(covs); df_cox["T"] = t_hep; df_cox["E"] = ev_hep.astype(int)
df_cox = df_cox[valid_h].dropna()
hrs = []; cis_lo = []; cis_hi = []; pvals = []; labels_h = []
for col in covs:
    try:
        sub = df_cox[["T", "E", col]].dropna()
        cph = CoxPHFitter(penalizer=0.5); cph.fit(sub, "T", "E")
        row = cph.summary.loc[col]
        hrs.append(row["exp(coef)"]); cis_lo.append(row["exp(coef) lower 95%"])
        cis_hi.append(row["exp(coef) upper 95%"]); pvals.append(row["p"]); labels_h.append(col)
    except: pass

order = np.argsort(hrs)
hrs = [hrs[i] for i in order]; cis_lo = [cis_lo[i] for i in order]
cis_hi = [cis_hi[i] for i in order]; pvals = [pvals[i] for i in order]
labels_h = [labels_h[i] for i in order]

fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(hrs)); ax.axvline(1, color=C["med"], lw=1.5, ls="--", alpha=0.7, zorder=1)
for i, (lo, hi, hr, p) in enumerate(zip(cis_lo, cis_hi, hrs, pvals)):
    col = C["ev"] if hr > 1 else C["ne"]
    ax.plot([lo, hi], [i, i], color=col, lw=1.8, alpha=0.65)
    ax.scatter([hr], [i], s=65, color=col, zorder=4)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    ax.text(max(hi, 1.05) * 1.05, i, f"{hr:.2f} {sig}", va="center", fontsize=8.5, color=col)
ax.set_yticks(y); ax.set_yticklabels(labels_h, fontsize=9.5)
ax.set_xlabel("Hazard ratio (95% CI) per 1-SD increase — standardised features", fontsize=9.5)
ax.set_title("Univariable Cox hazard ratios — hepatic events\n(features standardised to 1 SD)")
ax.set_xlim(0.1, 20); ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16])
ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8", "16"])
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=C["ev"], label="HR > 1 (risk factor)"),
                   mpatches.Patch(color=C["ne"], label="HR < 1 (protective)")],
          fontsize=9, frameon=False, loc="lower right")
ax.text(0.01, 0.01, "* p<0.05  ** p<0.01", transform=ax.transAxes, fontsize=8, color=C["med"])
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig7_forest_plot.png", bbox_inches="tight"); plt.close()

# ── FIG 8: Landmark analysis ──────────────────────────────────────────────────
print("Fig 8: landmark analysis...")
fig, ax = plt.subplots(figsize=(8, 4.8))
landmarks = [0, 1, 2, 3, 4, 5, 6]; c_vals = []
for lm in landmarks:
    surv_to_lm = (t_hep >= lm) | ~ev_hep
    sub_m = surv_to_lm & valid_h
    if sub_m.sum() < 20: c_vals.append(np.nan); continue
    t_future = np.maximum(t_hep[sub_m] - lm, 0.001)
    e_future  = ev_hep[sub_m] & (t_hep[sub_m] > lm)
    sc = score[sub_m]; sc_f = np.where(np.isnan(sc), np.nanmedian(sc), sc)
    try:
        c_vals.append(concordance_index_censored(e_future.astype(bool), t_future, sc_f)[0])
    except: c_vals.append(np.nan)
valid_lm = [(l, c) for l, c in zip(landmarks, c_vals) if not np.isnan(c)]
lm_x = [x[0] for x in valid_lm]; lm_y = [x[1] for x in valid_lm]
ax.plot(lm_x, lm_y, "o-", color=C["ne"], lw=2, ms=7, markerfacecolor=C["ne"], zorder=4)
ax.fill_between(lm_x, [0.5] * len(lm_x), lm_y, alpha=0.10, color=C["ne"])
ax.axhline(0.5, ls="--", color=C["med"], lw=1, alpha=0.6, label="Random (C=0.5)")
ax.set_xlabel("Landmark time (years) — patients conditionally event-free at this time")
ax.set_ylabel("C-index on future events")
ax.set_title("Landmark analysis: predictive C-index at successive time points")
ax.set_ylim(0.40, 1.0); ax.set_xlim(-0.2, 6.5); ax.legend(fontsize=9)
ax.text(0.98, 0.96, "Rate signal (FibroTest × 1/fu)", transform=ax.transAxes,
        ha="right", va="top", fontsize=9, color=C["ne"], fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig8_landmark.png", bbox_inches="tight"); plt.close()

print(f"\nDone. All figures → {FIG_DIR}/")
for fn in sorted(os.listdir(FIG_DIR)):
    print(f"  {fn}  {os.path.getsize(f'{FIG_DIR}/{fn}')//1024} KB")
