# ==========================================================
# BÀI 3
# XẾP HẠNG NGÀNH ƯU TIÊN CHUYỂN ĐỔI SỐ
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# ĐỌC DỮ LIỆU
# ==========================================================

df = pd.read_csv("vietnam_sectors_2024.csv")

print("\n========== DỮ LIỆU GỐC ==========\n")
print(df.head())

# ==========================================================
# CHUẨN HÓA MIN-MAX
# ==========================================================

cols_good = [
    "growth_rate_2024_pct",
    "gdp_share_2024_pct",
    "spillover_coef_0_1",
    "export_billion_USD",
    "labor_million",
    "ai_readiness_0_100"
]

col_bad = "automation_risk_pct"

def norm_good(x):
    return (x - x.min()) / (x.max() - x.min())

def norm_bad(x):
    return (x.max() - x) / (x.max() - x.min())

Xg = df[cols_good].apply(norm_good)

Xb = norm_bad(
    df[col_bad]
)

# ==========================================================
# 3.4.1 MA TRẬN CHUẨN HÓA
# ==========================================================

norm_df = Xg.copy()

norm_df["risk_score"] = Xb

norm_df["sector_name_vi"] = df["sector_name_vi"]

print("\n========== MA TRẬN CHUẨN HÓA ==========\n")
print(norm_df)

# ==========================================================
# 3.4.2 PRIORITY MẶC ĐỊNH
# ==========================================================

w_good = np.array([
    0.15,
    0.15,
    0.20,
    0.15,
    0.10,
    0.20
])

w_risk = 0.15

priority = (
    Xg.values @ w_good
    +
    w_risk * Xb.values
)

df["Priority"] = priority

ranking = (
    df[
        ["sector_name_vi", "Priority"]
    ]
    .sort_values(
        "Priority",
        ascending=False
    )
)

print("\n========== XẾP HẠNG ==========\n")
print(ranking)

# ==========================================================
# TOP 3
# ==========================================================

print("\nTOP 3 NGÀNH ƯU TIÊN\n")

print(
    ranking.head(3)
)

# ==========================================================
# 3.4.3 ĐỘ NHẠY AI READINESS
# ==========================================================

ai_weights = np.arange(
    0.05,
    0.45,
    0.05
)

heatmap_data = []

for ai_w in ai_weights:

    base = np.array([
        0.15,
        0.15,
        0.20,
        0.15,
        0.10,
        ai_w,
        0.15
    ])

    base = base / base.sum()

    w_good_new = base[:6]

    w_risk_new = base[6]

    score = (
        Xg.values @ w_good_new
        +
        w_risk_new * Xb.values
    )

    rank = pd.DataFrame({
        "Sector": df["sector_name_vi"],
        "Score": score
    })

    rank = rank.sort_values(
        "Score",
        ascending=False
    )

    top3 = (
        rank
        .head(3)
        ["Sector"]
        .tolist()
    )

    heatmap_data.append(top3)

    print(
        f"\nAI Weight = {ai_w:.2f}"
    )

    print(top3)

# ==========================================================
# HEATMAP
# ==========================================================

sector_list = (
    df["sector_name_vi"]
    .tolist()
)

heat = pd.DataFrame(
    0,
    index=sector_list,
    columns=[
        f"{w:.2f}"
        for w in ai_weights
    ]
)

for col, top3 in zip(
    heat.columns,
    heatmap_data
):
    for sector in top3:
        heat.loc[
            sector,
            col
        ] = 1

plt.figure(
    figsize=(10,6)
)

plt.imshow(
    heat,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(heat.columns)),
    heat.columns
)

plt.yticks(
    range(len(heat.index)),
    heat.index
)

plt.title(
    "Top-3 Sensitivity to AI Weight"
)

plt.tight_layout()

plt.savefig(
    "Heatmap_AI_Sensitivity.png",
    dpi=300
)

plt.show()

# ==========================================================
# 3.4.4 SO SÁNH HAI KỊCH BẢN
# ==========================================================

def compute_priority(
    good_weights,
    risk_weight
):

    score = (
        Xg.values @ good_weights
        +
        risk_weight * Xb.values
    )

    out = pd.DataFrame({
        "Sector": df["sector_name_vi"],
        "Priority": score
    })

    return out.sort_values(
        "Priority",
        ascending=False
    )

# ---------------------------
# ĐỊNH HƯỚNG TĂNG TRƯỞNG
# ---------------------------

growth_weights = np.array([
    0.25,   # growth
    0.20,   # productivity
    0.10,   # spillover
    0.20,   # export
    0.05,   # labor
    0.10    # AI
])

growth_risk = 0.10

growth_rank = compute_priority(
    growth_weights,
    growth_risk
)

# ---------------------------
# ĐỊNH HƯỚNG BAO TRÙM
# ---------------------------

inclusive_weights = np.array([
    0.10,   # growth
    0.10,   # productivity
    0.25,   # spillover
    0.05,   # export
    0.20,   # labor
    0.10    # AI
])

inclusive_risk = 0.20

inclusive_rank = compute_priority(
    inclusive_weights,
    inclusive_risk
)

print(
    "\n========== TĂNG TRƯỞNG ==========\n"
)

print(
    growth_rank.head(3)
)

print(
    "\n========== BAO TRÙM ==========\n"
)

print(
    inclusive_rank.head(3)
)

# ==========================================================
# XUẤT EXCEL
# ==========================================================

with pd.ExcelWriter(
    "Bai3_Result.xlsx",
    engine="openpyxl"
) as writer:

    norm_df.to_excel(
        writer,
        sheet_name="Normalized",
        index=False
    )

    ranking.to_excel(
        writer,
        sheet_name="Default_Ranking",
        index=False
    )

    growth_rank.to_excel(
        writer,
        sheet_name="Growth_Orientation",
        index=False
    )

    inclusive_rank.to_excel(
        writer,
        sheet_name="Inclusive_Orientation",
        index=False
    )

print(
    "\nĐã tạo Bai3_Result.xlsx"
)