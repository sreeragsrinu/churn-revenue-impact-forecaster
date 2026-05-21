# ============================================
# 📉 Churn Revenue Impact Forecaster 
# ============================================
# Developer : Sreerag

# ──────────────────────────────────────────
# CONFIG  ← one place for all settings
# ──────────────────────────────────────────
DATA_PATH = r"C:\Users\sreer\OneDrive\Desktop\Churn Revenue Impact Forecaster\data\Telco-Customer-Churn.csv"
OUTPUT_DIR       = r"C:\Users\sreer\OneDrive\Desktop\Churn Revenue Impact Forecaster\images"
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
N_ESTIMATORS     = 300
USE_SMOTE        = True
CLV_MONTHLY_CAP  = 24       # months to project CLV
DISCOUNT_RATE    = 0.10     # annual discount rate
RETENTION_COST   = 50       # $ per intervention
HIGH_RISK_THRESH = 0.60     # churn prob threshold for "high risk"
DPI              = 300      # chart resolution

# ──────────────────────────────────────────
# PALETTE  (used everywhere)
# ──────────────────────────────────────────
C_CHURN    = "#E05A40"   # red-orange  — churned
C_RETAIN   = "#3A86FF"   # cobalt blue — retained
C_ACCENT   = "#FFBE0B"   # amber       — highlights
C_NEUTRAL  = "#8B8FA8"   # muted grey
C_BG       = "#0F1117"   # chart background
C_PANEL    = "#181C27"   # axes background
C_GRID     = "#252A3A"   # grid lines
C_TEXT     = "#E2E6F0"   # primary text
C_SUBTEXT  = "#7B82A0"   # secondary text
PALETTE    = [C_RETAIN, C_CHURN, "#8338EC", C_ACCENT, "#06D6A0", "#FB5607"]

# ──────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.metrics           import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost          import XGBClassifier
import shap

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    warnings.warn("imbalanced-learn not installed — SMOTE disabled.")

warnings.filterwarnings("ignore")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# ──────────────────────────────────────────
# STYLE SETUP
# ──────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        "figure.facecolor":    C_BG,
        "axes.facecolor":      C_PANEL,
        "axes.edgecolor":      C_GRID,
        "axes.labelcolor":     C_TEXT,
        "axes.titlecolor":     C_TEXT,
        "axes.titlesize":      14,
        "axes.titleweight":    "bold",
        "axes.labelsize":      11,
        "axes.grid":           True,
        "grid.color":          C_GRID,
        "grid.linewidth":      0.6,
        "grid.alpha":          1.0,
        "xtick.color":         C_SUBTEXT,
        "ytick.color":         C_SUBTEXT,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "legend.facecolor":    C_PANEL,
        "legend.edgecolor":    C_GRID,
        "legend.labelcolor":   C_TEXT,
        "legend.fontsize":     10,
        "text.color":          C_TEXT,
        "figure.dpi":          100,
        "savefig.dpi":         DPI,
        "savefig.facecolor":   C_BG,
        "font.family":         "DejaVu Sans",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.spines.left":    False,
        "axes.spines.bottom":  False,
    })

apply_style()

def styled_fig(w=8, h=5, title=""):
    """Return a pre-styled figure + axes with named window title."""
    fig, ax = plt.subplots(figsize=(w, h))
    if title:
        fig.canvas.manager.set_window_title(title)
    return fig, ax

def add_watermark(fig):
    fig.text(0.99, 0.01, "Churn Revenue Impact Forecaster  v3",
             ha="right", va="bottom", fontsize=7, color=C_SUBTEXT, alpha=0.6)

def save_chart(fig, name, tight=True):
    add_watermark(fig)
    # Set the OS window title bar to the chart name (strip number prefix + extension)
    window_title = name.replace("_", " ").replace(".png", "").split(" ", 1)[-1].strip().title()
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, bbox_inches="tight" if tight else None)
    plt.show()
    plt.close(fig)
    print(f"  ✓ {name}")

def fmt_dollar(x, _): return f"${x:,.0f}"
def fmt_pct(x, _):    return f"{x:.0f}%"

# ──────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────
print("\n── 1. Loading data ──────────────────────")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}  |  {len(df.columns)} columns")

# ──────────────────────────────────────────
# 2. CLEAN
# ──────────────────────────────────────────
print("\n── 2. Cleaning ──────────────────────────")
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_null = df["TotalCharges"].isna().sum()
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)
print(f"   TotalCharges: {n_null} nulls → median-filled")

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
churn_rate = df["Churn"].mean()
print(f"   Churn rate: {churn_rate:.1%}  (imbalance {(1-churn_rate)/churn_rate:.1f}:1)")

# ──────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────
print("\n── 3. Feature engineering ───────────────")
df["TotalRevenue"]       = df["MonthlyCharges"] * df["tenure"]
df["AvgRevenuePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

monthly_rate    = DISCOUNT_RATE / 12
discount_factor = (1 - (1 + monthly_rate) ** -CLV_MONTHLY_CAP) / monthly_rate
df["CLV"]              = df["MonthlyCharges"] * discount_factor
df["ChargesPerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
df["HasMultiServices"] = (
    (df.get("PhoneService",    pd.Series("No", index=df.index)) == "Yes").astype(int) +
    (df.get("InternetService", pd.Series("No", index=df.index)) != "No").astype(int)
)
print(f"   CLV range: ${df['CLV'].min():.0f} – ${df['CLV'].max():.0f}  "
      f"({CLV_MONTHLY_CAP}mo @ {DISCOUNT_RATE:.0%} discount)")

# ──────────────────────────────────────────
# 4. ENCODE
# ──────────────────────────────────────────
print("\n── 4. Encoding ──────────────────────────")
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str)

df_enc = pd.get_dummies(df, drop_first=True)
nan_cols = df_enc.columns[df_enc.isna().any()].tolist()
if nan_cols:
    print(f"   Imputing {len(nan_cols)} NaN column(s)")
df_enc = df_enc.fillna(df_enc.median(numeric_only=True))
print(f"   Features after encoding: {df_enc.shape[1] - 1}")

# ──────────────────────────────────────────
# 5. EDA CHARTS
# ──────────────────────────────────────────
print("\n── 5. EDA charts ────────────────────────")

# ── Chart 01: Churn distribution (donut + bar side-by-side) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.canvas.manager.set_window_title("Customer Churn Overview")
fig.suptitle("Customer Churn Overview", fontsize=16, fontweight="bold",
             color=C_TEXT, y=0.92)
fig.subplots_adjust(top=0.82)

counts = df["Churn"].value_counts().sort_index()
labels = ["Retained", "Churned"]
colors = [C_RETAIN, C_CHURN]

wedges, texts, autotexts = ax1.pie(
    counts, labels=None, colors=colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor=C_BG, linewidth=2),
)
for at in autotexts:
    at.set(color=C_TEXT, fontsize=12, fontweight="bold")
ax1.legend(wedges, [f"{l}  ({v:,})" for l, v in zip(labels, counts)],
           loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2)
ax1.set_title("Distribution", fontsize=12, pad=12)

bars = ax2.bar(labels, counts.values, color=colors, width=0.45,
               edgecolor="none", zorder=3)
for bar, val in zip(bars, counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val:,}\n({val/len(df):.1%})",
             ha="center", va="bottom", fontsize=11, color=C_TEXT, fontweight="bold")
ax2.set_ylabel("Customers")
ax2.set_title("Count", fontsize=12, pad=12)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.set_ylim(0, counts.max() * 1.2)
ax2.tick_params(bottom=False)

plt.tight_layout()
save_chart(fig, "01_churn_overview.png")

# ── Chart 02: Monthly charges KDE overlay ──
fig, ax = styled_fig(9, 5)
ax.set_title("Monthly Charges  ·  Churned vs Retained", pad=14)
for label, color, lbl in [(0, C_RETAIN, "Retained"), (1, C_CHURN, "Churned")]:
    data = df.loc[df["Churn"] == label, "MonthlyCharges"]
    ax.hist(data, bins=35, alpha=0.35, color=color, edgecolor="none", density=True)
    data.plot.kde(ax=ax, color=color, lw=2.5, label=f"{lbl}  (μ=${data.mean():.0f})")
    ax.axvline(data.mean(), color=color, lw=1.2, ls="--", alpha=0.7)

ax.set_xlabel("Monthly Charges ($)")
ax.set_ylabel("Density")
ax.legend()
save_chart(fig, "02_monthly_charges.png")

# ── Chart 03: Tenure distribution ──
fig, ax = styled_fig(9, 5)
ax.set_title("Customer Tenure  ·  Churned vs Retained", pad=14)
for label, color, lbl in [(0, C_RETAIN, "Retained"), (1, C_CHURN, "Churned")]:
    data = df.loc[df["Churn"] == label, "tenure"]
    ax.hist(data, bins=24, alpha=0.35, color=color, edgecolor="none", density=True)
    data.plot.kde(ax=ax, color=color, lw=2.5, label=f"{lbl}  (μ={data.mean():.0f} mo)")
    ax.axvline(data.mean(), color=color, lw=1.2, ls="--", alpha=0.7)
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Density")
ax.legend()
save_chart(fig, "03_tenure.png")

# ── Chart 04: Churn rate by contract type (lollipop) ──
if "Contract" in df.columns:
    fig, ax = styled_fig(8, 4)
    ax.set_title("Churn Rate by Contract Type", pad=14)
    cr = df.groupby("Contract")["Churn"].mean().sort_values() * 100
    colors_lol = [C_RETAIN if v < 20 else C_ACCENT if v < 45 else C_CHURN for v in cr.values]
    ax.barh(cr.index, cr.values, color=colors_lol, height=0.08, zorder=3)
    ax.scatter(cr.values, cr.index, color=colors_lol, s=180, zorder=4)
    for y, v in zip(cr.index, cr.values):
        ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=11,
                fontweight="bold", color=C_TEXT)
    ax.set_xlabel("Churn Rate (%)")
    ax.set_xlim(0, cr.max() * 1.25)
    ax.grid(axis="y", alpha=0)
    save_chart(fig, "04_churn_by_contract.png")

# ── Chart 05: Correlation heatmap ──
fig, ax = plt.subplots(figsize=(11, 9))
fig.canvas.manager.set_window_title("Feature Correlation Heatmap")
top_feats = df_enc.corr()["Churn"].abs().sort_values(ascending=False)[1:16].index.tolist()
corr_sub  = df_enc[top_feats + ["Churn"]].corr()
mask      = np.triu(np.ones_like(corr_sub, dtype=bool), k=1)
cmap      = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(
    corr_sub, mask=mask, cmap=cmap, vmax=0.6, vmin=-0.6,
    center=0, annot=True, fmt=".2f", linewidths=0.4,
    linecolor=C_BG, ax=ax, annot_kws={"size": 8},
    cbar_kws={"shrink": 0.7},
)
ax.set_facecolor(C_PANEL)
ax.set_title("Feature Correlation Heatmap  (top 15 vs Churn)", pad=16,
             fontsize=14, fontweight="bold", color=C_TEXT)
ax.tick_params(colors=C_SUBTEXT)
plt.tight_layout()
save_chart(fig, "05_correlation_heatmap.png")

# ──────────────────────────────────────────
# 6. SPLIT
# ──────────────────────────────────────────
print("\n── 6. Train / test split ────────────────")
X = df_enc.drop("Churn", axis=1)
y = df_enc["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"   Train {X_train.shape[0]}  |  Test {X_test.shape[0]}")

# ──────────────────────────────────────────
# 7. SMOTE
# ──────────────────────────────────────────
if USE_SMOTE and SMOTE_AVAILABLE:
    print("\n── 7. SMOTE oversampling ────────────────")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_r, y_train_r = sm.fit_resample(X_train, y_train)
    print(f"   Resampled: {X_train_r.shape[0]} rows  (churn {y_train_r.mean():.1%})")
else:
    X_train_r, y_train_r = X_train, y_train

# ──────────────────────────────────────────
# 8. TRAIN
# ──────────────────────────────────────────
print("\n── 8. Training ──────────────────────────")
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

base_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced",
        max_features="sqrt", min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=0.05,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="logloss",
        random_state=RANDOM_STATE, verbosity=0),
}

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results  = {}

for name, base in base_models.items():
    print(f"\n   [{name}]")
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")
    model.fit(X_train_r, y_train_r)

    cv_auc = cross_val_score(model, X_train_r, y_train_r,
                             cv=cv_strat, scoring="roc_auc", n_jobs=-1)
    print(f"   CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    y_prob = model.predict_proba(X_test)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    f1s        = 2 * prec * rec / (prec + rec + 1e-9)
    best_thresh = thresholds[np.argmax(f1s[:-1])]
    y_pred      = (y_prob >= best_thresh).astype(int)

    results[name] = dict(
        model=model, y_prob=y_prob, y_pred=y_pred,
        threshold=best_thresh,
        accuracy  = accuracy_score(y_test, y_pred),
        roc_auc   = roc_auc_score(y_test, y_prob),
        avg_prec  = average_precision_score(y_test, y_prob),
        f1        = f1_score(y_test, y_pred),
        prec_curve=prec, rec_curve=rec,
    )
    print(f"   ROC-AUC {results[name]['roc_auc']:.4f}  |  "
          f"F1 {results[name]['f1']:.4f}  |  threshold {best_thresh:.3f}")

# ──────────────────────────────────────────
# 9. EVALUATION CHARTS
# ──────────────────────────────────────────
print("\n── 9. Evaluation charts ─────────────────")

model_colors = {"Random Forest": C_RETAIN, "XGBoost": C_CHURN}

# ── Chart 06: ROC curves ──
fig, ax = styled_fig(7, 6)
ax.set_title("ROC Curves  ·  Model Comparison", pad=14)
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    ax.plot(fpr, tpr, lw=2.5, color=model_colors[name],
            label=f"{name}  (AUC = {r['roc_auc']:.3f})")
    # mark operating threshold
    thresh_idx = np.argmin(np.abs(
        np.array([r["y_prob"]]) - r["threshold"]))
ax.plot([0,1],[0,1], "--", color=C_NEUTRAL, lw=1.2, label="Random chance")
ax.fill_between([0,1],[0,1],[0,0], alpha=0.04, color=C_NEUTRAL)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
save_chart(fig, "06_roc_curves.png")

# ── Chart 07: Precision-recall curves ──
fig, ax = styled_fig(7, 6)
ax.set_title("Precision-Recall Curves", pad=14)
for name, r in results.items():
    ax.plot(r["rec_curve"], r["prec_curve"], lw=2.5,
            color=model_colors[name],
            label=f"{name}  (AP = {r['avg_prec']:.3f})")
baseline = y_test.mean()
ax.axhline(baseline, ls="--", color=C_NEUTRAL, lw=1.2,
           label=f"Baseline (={baseline:.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
save_chart(fig, "07_precision_recall.png")

# ── Chart 08: Confusion matrices (annotated with %) ──
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.canvas.manager.set_window_title("Confusion Matrices · Test Set")
fig.suptitle("Confusion Matrices  ·  Test Set", fontsize=15,
             fontweight="bold", color=C_TEXT, y=1.02)

for ax, (name, r) in zip(axes, results.items()):
    cm   = confusion_matrix(y_test, r["y_pred"])
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "churn_cm", [C_PANEL, model_colors[name]], N=256)
    sns.heatmap(cm, annot=False, cmap=cmap, ax=ax, cbar=False,
                linewidths=1.5, linecolor=C_BG)
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.38, f"{cm[i,j]:,}",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold", color=C_TEXT)
            ax.text(j+0.5, i+0.62, f"({cm_n[i,j]:.1f}%)",
                    ha="center", va="center",
                    fontsize=11, color=C_TEXT, alpha=0.75)
    ax.set_xticklabels(["Retained", "Churned"], color=C_SUBTEXT)
    ax.set_yticklabels(["Retained", "Churned"], color=C_SUBTEXT, rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name}  ·  threshold={r['threshold']:.2f}",
                 fontsize=12, pad=10)

plt.tight_layout(rect=[0, 0, 1, 0.90])
save_chart(fig, "08_confusion_matrices.png")

# ── Chart 09: Feature importance ──
print("\n── 10. Feature importance ───────────────")
xgb_cal  = results["XGBoost"]["model"]
base_est = [cc.estimator for cc in xgb_cal.calibrated_classifiers_]
imps     = np.mean([e.feature_importances_ for e in base_est], axis=0)

imp_df = (pd.DataFrame({"Feature": X.columns, "Importance": imps})
          .sort_values("Importance", ascending=False)
          .head(15).reset_index(drop=True))

fig, ax = styled_fig(10, 7)
ax.set_title("Top 15 Feature Importances  ·  XGBoost", pad=14)

norm = plt.Normalize(imp_df["Importance"].min(), imp_df["Importance"].max())
cmap_imp = mcolors.LinearSegmentedColormap.from_list(
    "imp", ["#1E3A5F", C_RETAIN, C_ACCENT])

for i, row in imp_df[::-1].reset_index(drop=True).iterrows():
    color = cmap_imp(norm(row["Importance"]))
    bar   = ax.barh(i, row["Importance"], color=color,
                    height=0.65, edgecolor="none", zorder=3)
    ax.text(row["Importance"] + 0.001, i,
            f'{row["Importance"]:.4f}',
            va="center", fontsize=9.5, color=C_TEXT)

ax.set_yticks(range(len(imp_df)))
ax.set_yticklabels(imp_df["Feature"][::-1], fontsize=10)
ax.set_xlabel("Importance Score")
ax.set_xlim(0, imp_df["Importance"].max() * 1.18)
ax.grid(axis="y", alpha=0)
save_chart(fig, "09_feature_importance.png")

# ──────────────────────────────────────────
# SHAP Explainability
# ──────────────────────────────────────────
print("\n── SHAP explainability ─────────────────")

# Use one calibrated XGBoost estimator
xgb_model = base_est[0]

# Create explainer
explainer = shap.TreeExplainer(xgb_model)

# Sample data for speed
sample_size = min(300, len(X_test))
X_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)

# Calculate SHAP values
shap_values = explainer.shap_values(X_sample)

# SHAP Summary Plot
plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.title("SHAP Feature Impact")
plt.tight_layout()

shap_path = os.path.join(
    CHART_DIR,
    "13_shap_summary.png"
)

plt.savefig(
    shap_path,
    dpi=DPI,
    bbox_inches="tight",
    facecolor=C_BG
)

plt.show()
plt.close()

print("  ✓ 13_shap_summary.png")

# ── Chart 10: Churn probability histogram ──
best_name = max(results, key=lambda n: results[n]["roc_auc"])
best      = results[best_name]

fig, ax = styled_fig(9, 5)
ax.set_title(f"Churn Probability Distribution  ·  {best_name}", pad=14)
for label, color, lbl in [(0, C_RETAIN, "Actual retained"), (1, C_CHURN, "Actual churned")]:
    mask = y_test.values == label
    ax.hist(best["y_prob"][mask], bins=40, alpha=0.55, color=color,
            edgecolor="none", density=True, label=lbl)
ax.axvline(best["threshold"], color=C_ACCENT, lw=2, ls="--",
           label=f"Decision threshold ({best['threshold']:.2f})")
ax.set_xlabel("Predicted Churn Probability")
ax.set_ylabel("Density")
ax.legend()
save_chart(fig, "10_churn_probability_dist.png")

# ──────────────────────────────────────────
# 11. REVENUE IMPACT
# ──────────────────────────────────────────
print("\n── 11. Revenue impact ───────────────────")
print(f"   Best model: {best_name}  (ROC-AUC {best['roc_auc']:.4f})")

forecast = X_test.copy()
forecast["ActualChurn"]      = y_test.values
forecast["PredictedChurn"]   = best["y_pred"]
forecast["ChurnProbability"] = best["y_prob"]
forecast["CLV"]              = (
    df_enc["CLV"].loc[X_test.index].values
    if "CLV" in df_enc.columns else np.zeros(len(X_test))
)
forecast["ExpectedRevenueLoss"] = forecast["ChurnProbability"] * forecast["CLV"]
forecast["RetentionROI"]        = np.where(
    forecast["PredictedChurn"] == 1,
    forecast["ChurnProbability"] * forecast["CLV"] - RETENTION_COST, 0
)

# 4-quadrant segmentation (data-driven CLV threshold)
clv_median = forecast["CLV"].median()
def segment(row):
    hr = row["ChurnProbability"] >= HIGH_RISK_THRESH
    hv = row["CLV"] >= clv_median
    if hr and hv:  return "High risk · High value"
    if hr:         return "High risk · Low value"
    if hv:         return "Low risk · High value"
    return "Low risk · Low value"

forecast["RiskSegment"] = forecast.apply(segment, axis=1)

seg = (forecast.groupby("RiskSegment")
       .agg(Customers=("ChurnProbability","count"),
            AvgChurnProb=("ChurnProbability","mean"),
            TotalExpLoss=("ExpectedRevenueLoss","sum"),
            TotalROI=("RetentionROI","sum"))
       .reset_index().sort_values("TotalExpLoss", ascending=False))

# ── Chart 11: Revenue loss by segment ──
fig, ax = styled_fig(9, 5)
ax.set_title("Expected Revenue Loss by Risk Segment", pad=14)

seg_colors = [C_CHURN, "#F5A97F", C_RETAIN, "#A8C8F0"]
bars = ax.bar(seg["RiskSegment"], seg["TotalExpLoss"],
              color=seg_colors[:len(seg)], width=0.55, edgecolor="none", zorder=3)
for bar, row in zip(bars, seg.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f"${row.TotalExpLoss:,.0f}\n({row.Customers} customers)",
            ha="center", fontsize=10, color=C_TEXT, fontweight="bold")
ax.set_ylabel("Expected Revenue Loss ($)")
ax.yaxis.set_major_formatter(FuncFormatter(fmt_dollar))
ax.set_ylim(0, seg["TotalExpLoss"].max() * 1.25)
ax.tick_params(axis="x", labelsize=9)
save_chart(fig, "11_revenue_loss_by_segment.png")

# ── Chart 12: Retention ROI scatter ──
fig, ax = styled_fig(9, 7)
ax.set_title("Retention ROI  ·  CLV vs Churn Probability\n"
             "(bubble size = expected revenue loss)", pad=14)

plot_df  = forecast[forecast["PredictedChurn"] == 1].copy()
scatter  = ax.scatter(
    plot_df["ChurnProbability"],
    plot_df["CLV"],
    c=plot_df["RetentionROI"],
    s=np.clip(plot_df["ExpectedRevenueLoss"] / 8, 20, 300),
    cmap=mcolors.LinearSegmentedColormap.from_list(
        "roi", [C_NEUTRAL, C_ACCENT, C_CHURN]),
    alpha=0.65, edgecolors="none", zorder=3
)
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Retention ROI ($)", color=C_TEXT, fontsize=10)
cbar.ax.yaxis.set_tick_params(color=C_SUBTEXT)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_SUBTEXT)

ax.axvline(HIGH_RISK_THRESH, color=C_ACCENT, lw=1.2, ls="--", alpha=0.6,
           label=f"High risk threshold ({HIGH_RISK_THRESH})")
ax.axhline(clv_median, color=C_RETAIN, lw=1.2, ls="--", alpha=0.6,
           label=f"Median CLV (${clv_median:.0f})")
ax.set_xlabel("Churn Probability")
ax.set_ylabel("Customer Lifetime Value ($)")
ax.yaxis.set_major_formatter(FuncFormatter(fmt_dollar))
ax.legend(loc="upper left")
save_chart(fig, "12_retention_roi_scatter.png")

# ──────────────────────────────────────────
# 12. SUMMARY
# ──────────────────────────────────────────
print("\n── 12. Summary ──────────────────────────")
total_loss       = forecast["ExpectedRevenueLoss"].sum()
total_roi        = forecast.loc[forecast["RetentionROI"] > 0, "RetentionROI"].sum()
at_risk          = (forecast["PredictedChurn"] == 1).sum()

print(f"\n   {'Customers flagged as churners':<35}: {at_risk:,}")
print(f"   {'Total expected revenue loss':<35}: ${total_loss:,.2f}")
print(f"   {'Total retention ROI (if acted)':<35}: ${total_roi:,.2f}")
print(f"\n   Segment breakdown:")
print(seg.to_string(index=False))

print("\n   Model comparison:")
for name, r in results.items():
    print(f"   {name:<20} ROC-AUC {r['roc_auc']:.4f}  "
          f"F1 {r['f1']:.4f}  AP {r['avg_prec']:.4f}  "
          f"Accuracy {r['accuracy']:.4f}  threshold {r['threshold']:.3f}")

# ──────────────────────────────────────────
# 13. EXPORT
# ──────────────────────────────────────────
print("\n── 13. Exporting ────────────────────────")
out_csv = os.path.join(OUTPUT_DIR, "churn_revenue_forecast_results.csv")
forecast.to_csv(out_csv, index=False)
print(f"  ✓ {out_csv}")

model_csv = os.path.join(OUTPUT_DIR, "model_comparison.csv")
pd.DataFrame([
    dict(Model=n, ROC_AUC=f"{r['roc_auc']:.4f}", Avg_Prec=f"{r['avg_prec']:.4f}",
         F1=f"{r['f1']:.4f}", Accuracy=f"{r['accuracy']:.4f}",
         Threshold=f"{r['threshold']:.3f}")
    for n, r in results.items()
]).to_csv(model_csv, index=False)
print(f"  ✓ {model_csv}")

print(f"\n✅  Done.  12 charts + 2 CSVs saved to: {os.path.abspath(OUTPUT_DIR)}")