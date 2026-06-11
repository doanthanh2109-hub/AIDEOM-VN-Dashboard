import pulp
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# DATAS & PARAMETERS (DỮ LIỆU VÀ THAM SỐ)
# ==============================================================================
# Danh sách 6 vùng kinh tế - xã hội và 4 hạng mục đầu tư
regions = ['Trung du MN phía Bắc (NMM)', 'Đồng bằng sông Hồng (RRD)', 
           'Bắc Trung Bộ & duyên hải MT (NCC)', 'Tây Nguyên (CH)', 
           'Đông Nam Bộ (SE)', 'Đồng bằng sông Cửu Long (MD)']

items = ['I', 'D', 'AI', 'H']
# I: Hạ tầng số, D: Chuyển đổi số DN, AI: Năng lực AI, H: Nhân lực số

n_regions = len(regions)
n_items = len(items)

# Bảng hệ số tác động biên beta [r, j] (Đơn vị: nghìn VND GDP / 1 triệu VND đầu tư)
# Thứ tự các cột tương ứng với: I, D, AI, H
beta_matrix = np.array([
    [1.15, 0.85, 0.55, 1.30],  # Vùng 1: Trung du miền núi phía Bắc
    [0.95, 1.25, 1.40, 1.05],  # Vùng 2: Đồng bằng sông Hồng
    [1.05, 0.95, 0.85, 1.15],  # Vùng 3: Bắc Trung Bộ + DH Trung Bộ
    [1.20, 0.75, 0.45, 1.35],  # Vùng 4: Tây Nguyên
    [0.90, 1.30, 1.55, 1.00],  # Vùng 5: Đông Nam Bộ
    [1.10, 0.85, 0.65, 1.25]   # Vùng 6: Đồng bằng sông Cửu Long
])

# Chỉ số số hóa ban đầu D_r của 6 vùng
D_init = np.array([38.0, 78.0, 55.0, 32.0, 82.0, 48.0])

# Các hằng số chính sách
TOTAL_BUDGET = 50000.0  # (C1) Tổng ngân sách: 50,000 tỷ VND
MIN_REG_BUDGET = 5000.0 # (C2) Sàn ngân sách mỗi vùng: 5,000 tỷ VND
MAX_REG_BUDGET = 12000.0# (C3) Trần ngân sách mỗi vùng: 12,000 tỷ VND
MIN_H_BUDGET = 12000.0  # (C4) Sàn nhân lực số tổng: 12,000 tỷ VND
GAMMA = 0.002           # (C5) Hệ số tác động số hóa gamma
LAMBDA = 0.7            # (C5) Hệ số công bằng vùng miền lambda

# ==============================================================================
# CÂU 4.4.1: CÀI ĐẶT VÀ GIẢI MÔ HÌNH BẰNG PuLP (CÓ RÀNG BUỘC CÔNG BẰNG C5)
# ==============================================================================
print("-" * 60)
print("CÂU 4.4.1: GIẢI MÔ HÌNH BẰNG PuLP (CÓ RÀNG BUỘC CÔNG BẰNG C5)")
print("-" * 60)

# Khởi tạo bài toán Tối đa hóa
prob_pulp = pulp.LpProblem("Budget_Allocation_With_Fairness", pulp.LpMaximize)

# Biến quyết định x[r, j]: lượng ngân sách đầu tư vào vùng r cho hạng mục j
x_pulp = pulp.LpVariable.dicts("x", ((r, j) for r in range(n_regions) for j in range(n_items)), lowBound=0)

# Biến bổ trợ M đại diện cho giá trị cực đại của biểu thức số hóa: max_r (D_r + gamma * x_D,r)
M = pulp.LpVariable("Max_Digital_Index")

# Hàm mục tiêu: Tối đa hóa tổng GDP Gain = sum(beta_rj * x_rj)
prob_pulp += pulp.lpSum(beta_matrix[r, j] * x_pulp[r, j] for r in range(n_regions) for j in range(n_items))

# Ràng buộc (C1): Tổng ngân sách toàn quốc <= 50,000 tỷ
prob_pulp += pulp.lpSum(x_pulp[r, j] for r in range(n_regions) for j in range(n_items)) <= TOTAL_BUDGET

# Ràng buộc (C2) & (C3): Sàn và Trần ngân sách đối với từng vùng
for r in range(n_regions):
    prob_pulp += pulp.lpSum(x_pulp[r, j] for j in range(n_items)) >= MIN_REG_BUDGET
    prob_pulp += pulp.lpSum(x_pulp[r, j] for j in range(n_items)) <= MAX_REG_BUDGET

# Ràng buộc (C4): Sàn đầu tư nhân lực số tổng quốc gia >= 12,000 tỷ (H tương ứng với index 3)
prob_pulp += pulp.lpSum(x_pulp[r, 3] for r in range(n_regions)) >= MIN_H_BUDGET

# Ràng buộc (C5) Công bằng vùng miền:
# B1: Định nghĩa M = max_r (D_r + gamma * x_D,r) -> M >= D_r + gamma * x_D,r với mọi r (D ứng với index 1)
for r in range(n_regions):
    prob_pulp += M >= D_init[r] + GAMMA * x_pulp[r, 1]

# B2: Đảm bảo mọi vùng đều đạt tối thiểu lambda * M
for r in range(n_regions):
    prob_pulp += D_init[r] + GAMMA * x_pulp[r, 1] >= LAMBDA * M

# Giải mô hình bằng CBC Solver mặc định
status_pulp = prob_pulp.solve(pulp.PULP_CBC_CMD(msg=False))

# Trích xuất kết quả tối ưu từ PuLP
pulp_results = np.zeros((n_regions, n_items))
for r in range(n_regions):
    for j in range(n_items):
        pulp_results[r, j] = x_pulp[r, j].varValue

Z_pulp = pulp.value(prob_pulp.objective)
print(f"Trạng thái giải: {pulp.LpStatus[status_pulp]}")
print(f"Giá trị mục tiêu tối ưu (Tổng GDP Gain) Z* = {Z_pulp:,.2f} tỷ VND")
print("\nMa trận phân bổ ngân sách tối ưu 6x4 (PuLP):")
df_pulp = pd.DataFrame(pulp_results, index=regions, columns=items)
print(df_pulp.round(2))


# ==============================================================================
# CÂU 4.4.2: CÀI ĐẶT VÀ GIẢI MÔ HÌNH BẰNG CVXPY ĐỂ ĐỐI CHIẾU
# ==============================================================================
print("\n" + "-" * 60)
print("CÂU 4.4.2: GIẢI MÔ HÌNH BẰNG CVXPY ĐỂ ĐỐI CHIẾU")
print("-" * 60)

# Định nghĩa biến quyết định trong CVXPY (Ma trận kích thước 6x4, không âm)
x_cvx = cp.Variable((n_regions, n_items), nonneg=True)

# Hàm mục tiêu: Tối đa hóa tổng tích chập các phần tử (Element-wise multiplication)
obj_cvx = cp.Maximize(cp.sum(cp.multiply(beta_matrix, x_cvx)))

# Danh sách ràng buộc
constraints_cvx = [
    cp.sum(x_cvx) <= TOTAL_BUDGET,                       # (C1)
    cp.sum(x_cvx, axis=1) >= MIN_REG_BUDGET,             # (C2)
    cp.sum(x_cvx, axis=1) <= MAX_REG_BUDGET,             # (C3)
    cp.sum(x_cvx[:, 3]) >= MIN_H_BUDGET                  # (C4)
]

# (C5) Ràng buộc công bằng vùng miền: biểu diễn trực tiếp qua hàm cp.max
D_post_cvx = D_init + GAMMA * x_cvx[:, 1]
constraints_cvx.append(D_post_cvx >= LAMBDA * cp.max(D_post_cvx))

# Tiến hành giải toán
prob_cvx = cp.Problem(obj_cvx, constraints_cvx)
prob_cvx.solve(solver=cp.CLARABEL)# Sử dụng ECOS solver cho bài toán LP tuyến tính

Z_cvx = prob_cvx.value
print(f"Giá trị mục tiêu tối ưu từ CVXPY Z* = {Z_cvx:,.2f} tỷ VND")
print(f"Chênh lệch kết quả giữa PuLP và CVXPY: {abs(Z_pulp - Z_cvx):.4f} tỷ VND")


# ==============================================================================
# CÂU 4.4.4: GIẢI MÔ HÌNH KHÔNG CÓ RÀNG BUỘC CÔNG BẰNG (BỎ C5) ĐỂ ĐỐI CHIẾU CHI PHÍ
# ==============================================================================
print("\n" + "-" * 60)
print("CÂU 4.4.4: SO SÁNH VỚI MÔ HÌNH KHÔNG CÓ CÔNG BẰNG (BỎ C5)")
print("-" * 60)

prob_unfair = pulp.LpProblem("Budget_Allocation_Unfair", pulp.LpMaximize)
x_unfair = pulp.LpVariable.dicts("x_un", ((r, j) for r in range(n_regions) for j in range(n_items)), lowBound=0)

prob_unfair += pulp.lpSum(beta_matrix[r, j] * x_unfair[r, j] for r in range(n_regions) for j in range(n_items))
prob_unfair += pulp.lpSum(x_unfair[r, j] for r in range(n_regions) for j in range(n_items)) <= TOTAL_BUDGET

for r in range(n_regions):
    prob_unfair += pulp.lpSum(x_unfair[r, j] for j in range(n_items)) >= MIN_REG_BUDGET
    prob_unfair += pulp.lpSum(x_unfair[r, j] for j in range(n_items)) <= MAX_REG_BUDGET

prob_unfair += pulp.lpSum(x_unfair[r, 3] for r in range(n_regions)) >= MIN_H_BUDGET

prob_unfair.solve(pulp.PULP_CBC_CMD(msg=False))
Z_unfair = pulp.value(prob_unfair.objective)

pulp_unfair_results = np.zeros((n_regions, n_items))
for r in range(n_regions):
    for j in range(n_items):
        pulp_unfair_results[r, j] = x_unfair[r, j].varValue

print(f"Z* khi CÓ ràng buộc công bằng (C5): {Z_pulp:,.2f} tỷ VND")
print(f"Z* khi KHÔNG CÓ ràng buộc công bằng (Bỏ C5): {Z_unfair:,.2f} tỷ VND")
print(f"==> Chi phí kinh tế đánh đổi cho sự công bằng vùng miền: {Z_unfair - Z_pulp:,.2f} tỷ VND GDP Gain.")


# ==============================================================================
# CÂU 4.4.3: TRỰC QUAN HÓA KẾT QUẢ PHÂN BỔ (HEATMAP)
# ==============================================================================
# Thiết lập cấu hình hiển thị tiếng Việt cho đồ thị matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 6)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Heatmap Kịch bản có công bằng vùng miền
sns.heatmap(df_pulp, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5, ax=axes[0])
axes[0].set_title("Phân bổ ngân sách CÓ RÀNG BUỘC CÔNG BẰNG C5 (tỷ VND)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Hạng mục đầu tư", fontsize=10)
axes[0].set_ylabel("Vùng kinh tế - xã hội", fontsize=10)

# Heatmap Kịch bản không có công bằng vùng miền
df_unfair = pd.DataFrame(pulp_unfair_results, index=regions, columns=items)
sns.heatmap(df_unfair, annot=True, fmt=".1f", cmap="OrRd", linewidths=.5, ax=axes[1])
axes[1].set_title("Phân bổ ngân sách KHÔNG CÓ RÀNG BUỘC CÔNG BẰNG (tỷ VND)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Hạng mục đầu tư", fontsize=10)
axes[1].set_ylabel("Vùng kinh tế - xã hội", fontsize=10)

plt.tight_layout()
print("\n[HỆ THỐNG] Đang hiển thị đồ thị Heatmap so sánh...")
plt.show()