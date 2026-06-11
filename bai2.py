# ==========================================================
# BÀI 2
# TỐI ƯU PHÂN BỔ NGÂN SÁCH CHUYỂN ĐỔI SỐ
# ==========================================================

from scipy.optimize import linprog
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# 2.4.1 GIẢI BÀI TOÁN BẰNG SCIPY
# ==========================================================

print("\n" + "="*60)
print("2.4.1 GIẢI BÀI TOÁN BẰNG SCIPY")
print("="*60)

# Max Z => Min (-Z)
c = [-0.85, -1.20, -0.95, -1.35]

A_ub = [
    [1, 1, 1, 1],          # Tổng ngân sách <= 100

    [-1, 0, 0, 0],         # x1 >= 25

    [0, -1, 0, 0],         # x2 >= 15

    [0, 0, -1, 0],         # x3 >= 20

    [0, 0, 0, -1],         # x4 >= 10

    [0.35, -0.65, 0.35, -0.65]
]

b_ub = [
    100,
    -25,
    -15,
    -20,
    -10,
    0
]

res = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=[(0, None)] * 4,
    method="highs"
)

print(res)

# ==========================================================
# HIỂN THỊ KẾT QUẢ TỐI ƯU
# ==========================================================

x1, x2, x3, x4 = res.x

Z = -res.fun

print("\nPHƯƠNG ÁN TỐI ƯU")

print(f"x1 (Hạ tầng số)      = {x1:.2f}")
print(f"x2 (AI & Dữ liệu)    = {x2:.2f}")
print(f"x3 (Nhân lực số)     = {x3:.2f}")
print(f"x4 (R&D công nghệ)   = {x4:.2f}")

print(f"\nGiá trị tối ưu Z = {Z:.2f}")

# ==========================================================
# 2.4.2 SHADOW PRICE (DUAL VALUE)
# ==========================================================

print("\n" + "="*60)
print("2.4.2 SHADOW PRICE")
print("="*60)

constraint_names = [
    "Tong ngan sach",
    "x1 >= 25",
    "x2 >= 15",
    "x3 >= 20",
    "x4 >= 10",
    "Rang buoc chien luoc"
]

for name, dual in zip(
    constraint_names,
    res.ineqlin.marginals
):
    print(
        f"{name:<25} : {dual:.6f}"
    )

# ==========================================================
# 2.4.3 PHÂN TÍCH ĐỘ NHẠY
# ==========================================================

print("\n" + "="*60)
print("2.4.3 PHÂN TÍCH ĐỘ NHẠY")
print("="*60)

budgets = [100, 120, 140]

z_values = []

for B in budgets:

    b_test = [
        B,
        -25,
        -15,
        -20,
        -10,
        0
    ]

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_test,
        bounds=[(0, None)] * 4,
        method="highs"
    )

    z = -result.fun

    z_values.append(z)

    print(
        f"Budget = {B:>3} --> Z = {z:.2f}"
    )

# ==========================================================
# BIỂU ĐỒ ĐỘ NHẠY
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    budgets,
    z_values,
    marker="o",
    linewidth=2
)

plt.title(
    "Sensitivity Analysis"
)

plt.xlabel(
    "Budget"
)

plt.ylabel(
    "Optimal GDP Gain"
)

plt.grid(True)

plt.savefig(
    "Sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# 2.4.4 THÊM RÀNG BUỘC x3 >= 30
# ==========================================================

print("\n" + "="*60)
print("2.4.4 ƯU TIÊN NHÂN LỰC SỐ")
print("="*60)

b_new = [
    100,
    -25,
    -15,
    -30,
    -10,
    0
]

res_new = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_new,
    bounds=[(0, None)] * 4,
    method="highs"
)

if res_new.success:

    print(
        "\nBài toán vẫn khả thi."
    )

    print(
        f"Z mới = {-res_new.fun:.2f}"
    )

    print(
        f"x1 = {res_new.x[0]:.2f}"
    )

    print(
        f"x2 = {res_new.x[1]:.2f}"
    )

    print(
        f"x3 = {res_new.x[2]:.2f}"
    )

    print(
        f"x4 = {res_new.x[3]:.2f}"
    )

else:

    print(
        "\nBài toán không còn khả thi."
    )

# ==========================================================
# XUẤT KẾT QUẢ RA EXCEL
# ==========================================================

result_df = pd.DataFrame({
    "Variable": [
        "x1",
        "x2",
        "x3",
        "x4"
    ],
    "Optimal_Value": [
        x1,
        x2,
        x3,
        x4
    ]
})

sensitivity_df = pd.DataFrame({
    "Budget": budgets,
    "Optimal_Z": z_values
})

with pd.ExcelWriter(
    "Bai2_Result.xlsx",
    engine="openpyxl"
) as writer:

    result_df.to_excel(
        writer,
        sheet_name="Optimal_Solution",
        index=False
    )

    sensitivity_df.to_excel(
        writer,
        sheet_name="Sensitivity",
        index=False
    )

print("\nĐã tạo file Bai2_Result.xlsx")
print("Đã lưu biểu đồ Sensitivity.png")