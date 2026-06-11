import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("vietnam_macro_2020_2025.csv")

Year = df["year"].values
Y = df["GDP_trillion_VND"].values

K = np.array([16500, 17800, 19600, 21300, 23500, 25900])

L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])

D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])

AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])

H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])

alpha = 0.33
beta = 0.42
gamma = 0.10
delta = 0.08
theta = 0.07

data = pd.DataFrame({
    "Year": Year,
    "Y": Y,
    "K": K,
    "L": L,
    "D": D,
    "AI": AI,
    "H": H
})
# ==========================================================
# 1.4.1 TÍNH TFP
# ==========================================================

data["A"] = Y / (
    (K ** alpha)
    * (L ** beta)
    * (D ** gamma)
    * (AI ** delta)
    * (H ** theta)
)

print("\n========== TFP ==========")
print(data[["Year", "A"]])

# Biểu đồ TFP
plt.figure(figsize=(8,5))
plt.plot(
    Year,
    data["A"],
    marker="o",
    linewidth=2
)

plt.title("Total Factor Productivity (TFP)")
plt.xlabel("Year")
plt.ylabel("A")
plt.grid(True)
plt.show()

# ==========================================================
# 1.4.2 GDP DỰ BÁO
# ==========================================================

A_bar = data["A"].mean()

data["GDP_pred"] = (
    A_bar
    * (K ** alpha)
    * (L ** beta)
    * (D ** gamma)
    * (AI ** delta)
    * (H ** theta)
)

data["APE"] = (
    np.abs(Y - data["GDP_pred"])
    / Y
) * 100

MAPE = data["APE"].mean()

print("\n========== GDP DỰ BÁO ==========")

print(
    data[
        [
            "Year",
            "Y",
            "GDP_pred",
            "APE"
        ]
    ]
)

print(f"\nMAPE = {MAPE:.4f}%")

# Biểu đồ GDP

plt.figure(figsize=(10,6))

plt.plot(
    Year,
    Y,
    marker="o",
    linewidth=2,
    label="Actual GDP"
)

plt.plot(
    Year,
    data["GDP_pred"],
    marker="s",
    linewidth=2,
    label="Predicted GDP"
)

plt.title("Actual GDP vs Predicted GDP")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.legend()
plt.grid(True)

plt.show()

# ==========================================================
# 1.4.3 GROWTH ACCOUNTING
# ==========================================================

growth = pd.DataFrame()

growth["Period"] = [
    f"{Year.iloc[i-1]}-{Year.iloc[i]}"
    for i in range(1, len(data))
]

growth["dlnY"] = np.diff(np.log(Y))

growth["Capital_K"] = alpha * np.diff(np.log(K))
growth["Labor_L"] = beta * np.diff(np.log(L))
growth["Digital_D"] = gamma * np.diff(np.log(D))
growth["AI"] = delta * np.diff(np.log(AI))
growth["Human_H"] = theta * np.diff(np.log(H))
growth["TFP"] = np.diff(np.log(data["A"]))

print("\n========== GROWTH ACCOUNTING ==========")
print(growth)

# Đóng góp bình quân

avg_contribution = {
    "Capital": growth["Capital_K"].mean(),
    "Labor": growth["Labor_L"].mean(),
    "Digital": growth["Digital_D"].mean(),
    "AI": growth["AI"].mean(),
    "Human": growth["Human_H"].mean(),
    "TFP": growth["TFP"].mean()
}

total = sum(avg_contribution.values())

contribution_df = pd.DataFrame({
    "Factor": avg_contribution.keys(),
    "Contribution (%)": [
        value / total * 100
        for value in avg_contribution.values()
    ]
})

print("\n========== ĐÓNG GÓP (%) ==========")
print(contribution_df)

# Biểu đồ đóng góp

plt.figure(figsize=(10,6))

plt.bar(
    contribution_df["Factor"],
    contribution_df["Contribution (%)"]
)

plt.title("Growth Contribution 2020-2025")
plt.ylabel("Contribution (%)")
plt.grid(axis="y")

plt.show()

# ==========================================================
# 1.4.4 DỰ BÁO GDP 2030
# ==========================================================

years_forward = 5

K_2030 = K.iloc[-1] * (1.06 ** years_forward)
L_2030 = L.iloc[-1] * (1.06 ** years_forward)

D_2030 = 30
AI_2030 = 100
H_2030 = 35

A_2030 = (
    data["A"].iloc[-1]
    * (1.012 ** years_forward)
)

GDP_2030 = (
    A_2030
    * (K_2030 ** alpha)
    * (L_2030 ** beta)
    * (D_2030 ** gamma)
    * (AI_2030 ** delta)
    * (H_2030 ** theta)
)

print("\n========== GDP 2030 ==========")

print(f"K_2030   = {K_2030:.2f}")
print(f"L_2030   = {L_2030:.2f}")
print(f"D_2030   = {D_2030}")
print(f"AI_2030  = {AI_2030}")
print(f"H_2030   = {H_2030}")
print(f"A_2030   = {A_2030:.4f}")

print("\n--------------------------------")
print(f"GDP_2030 = {GDP_2030:.2f}")
print("--------------------------------")

# ==========================================================
# XUẤT KẾT QUẢ
# ==========================================================

with pd.ExcelWriter(
    "Bai1_Result.xlsx",
    engine="openpyxl"
) as writer:

    data.to_excel(
        writer,
        sheet_name="Main_Data",
        index=False
    )

    growth.to_excel(
        writer,
        sheet_name="Growth_Accounting",
        index=False
    )

    contribution_df.to_excel(
        writer,
        sheet_name="Contribution",
        index=False
    )

print("\nĐã tạo file: Bai1_Result.xlsx")