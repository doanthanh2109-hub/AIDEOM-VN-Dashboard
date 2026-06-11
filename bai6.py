import pandas as pd
import pulp as pl
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu từ file csv vùng kinh tế
try:
    df_regions = pd.read_csv("vietnam_regions_2024.csv")
    print("Đọc dữ liệu file vietnam_regions_2024.csv thành công!")
except FileNotFoundError:
    print("Vui lòng đặt file vietnam_regions_2024.csv chung thư mục làm việc.")
    raise

regions = df_regions['region_name_vi'].tolist()

# 2. Khởi tạo các tham số mô hình
B = 15000       # Tổng ngân sách phân bổ (tỷ VND)
x_min = 1000    # Sàn đầu tư tối thiểu cho mỗi vùng (tỷ VND)

# Thiết lập các hệ số alpha, beta, gamma cho các mục tiêu dựa trên dữ liệu thực tế
alpha = {}  # Hiệu quả tăng trưởng kinh tế
beta = {}   # Tác động giảm bất bình đẳng (Gini cao -> cần ưu tiên giảm)
gamma = {}  # Hiệu quả nâng cao sẵn sàng AI

for idx, row in df_regions.iterrows():
    r = row['region_name_vi']
    # alpha: tỷ lệ thuận với quy mô kinh tế và tốc độ tăng trưởng hiện tại
    alpha[r] = (row['grdp_trillion_VND'] * (row['grdp_growth_pct'] / 100.0)) / 100.0
    # beta: tỷ lệ thuận với hệ số Gini (muốn đầu tư vào nơi có Gini cao để cân bằng xã hội)
    beta[r] = row['gini_coef']
    # gamma: tỷ lệ thuận với chỉ số sẵn sàng AI hiện tại để bứt phá công nghệ
    gamma[r] = row['ai_readiness_0_100'] / 100.0

# 3. Hàm phụ trợ để giải tìm giá trị Max/Min phục vụ chuẩn hóa (Payoff Matrix)
def solve_single_objective(target="growth", mode="max"):
    prob = pl.LpProblem("Single_Objective", pl.LpMaximize if mode == "max" else pl.LpMinimize)
    x = pl.LpVariable.dicts("x", regions, lowBound=x_min, cat='Continuous')
    
    # Ràng buộc ngân sách
    prob += pl.lpSum([x[r] for r in regions]) <= B
    
    # Hàm mục tiêu đơn lẻ
    if target == "growth":
        prob += pl.lpSum([alpha[r] * x[r] for r in regions])
    elif target == "gini":
        prob += pl.lpSum([beta[r] * x[r] for r in regions])
    elif target == "ai":
        prob += pl.lpSum([gamma[r] * x[r] for r in regions])
        
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    return pl.value(prob.objective)

# Tính toán các giá trị biên để chuẩn hóa (khử đơn vị đo lường khác nhau)
f1_max = solve_single_objective("growth", "max")
f1_min = solve_single_objective("growth", "min")

f2_max = solve_single_objective("gini", "max")
f2_min = solve_single_objective("gini", "min")

f3_max = solve_single_objective("ai", "max")
f3_min = solve_single_objective("ai", "min")

# 4. Xây dựng và giải mô hình Đa mục tiêu (Weighted Sum Method)
mo_prob = pl.LpProblem("Multi_Objective_Goal_Programming", pl.LpMaximize)
x_mo = pl.LpVariable.dicts("X_Digital_Budget", regions, lowBound=x_min, cat='Continuous')

# Ràng buộc giới hạn tổng ngân sách
mo_prob += pl.lpSum([x_mo[r] for r in regions]) <= B, "Total_Budget_Constraint"

# Định nghĩa các biểu thức mục tiêu gốc
f1_expr = pl.lpSum([alpha[r] * x_mo[r] for r in regions])
f2_expr = pl.lpSum([beta[r] * x_mo[r] for r in regions])
f3_expr = pl.lpSum([gamma[r] * x_mo[r] for r in regions])

# Trọng số ưu tiên chính sách
w1, w2, w3 = 0.4, 0.3, 0.3

# Hàm mục tiêu hỗn hợp chuẩn hóa: Maximize Kinh tế + Maximize Công nghệ - Minimize Bất bình đẳng
# Do Gini là hàm Minimize, khi đưa vào hàm Max chung ta đảo dấu thành trừ (-)
mo_prob += (
    w1 * ((f1_expr - f1_min) / (f1_max - f1_min if f1_max != f1_min else 1)) -
    w2 * ((f2_expr - f2_min) / (f2_max - f2_min if f2_max != f2_min else 1)) +
    w3 * ((f3_expr - f3_min) / (f3_max - f3_min if f3_max != f3_min else 1))
), "Combined_Social_Utility"

# Giải mô hình đa mục tiêu
mo_prob.solve(pl.PULP_CBC_CMD(msg=False))

# 5. Chiết xuất kết quả và hiển thị báo cáo dữ liệu
results = []
for r in regions:
    results.append({
        'Vùng Kinh Tế': r,
        'Ngân Sách Số Phân Bổ (Tỷ VND)': round(x_mo[r].varValue, 2)
    })
df_res = pd.DataFrame(results)

print(f"Trạng thái tối ưu đa mục tiêu: {pl.LpStatus[mo_prob.status]}")
print("\n--- BẢNG KẾT QUẢ PHÂN BỔ NGÂN SÁCH ĐA MỤC TIÊU THEO VÙNG ---")
print(df_res.to_string(index=False))

# Tính toán mức độ đạt được của từng mục tiêu đơn lẻ so với trạng thái lý tưởng của nó
final_f1 = sum(alpha[r] * x_mo[r].varValue for r in regions)
final_f2 = sum(beta[r] * x_mo[r].varValue for r in regions)
final_f3 = sum(gamma[r] * x_mo[r].varValue for r in regions)

print("\n--- ĐÁNH GIÁ MỨC ĐỘ ĐẠT MỤC TIÊU CHÍNH SÁCH ---")
print(f"1. Tổng GRDP gia tăng kinh tế: {round(final_f1, 2)} (Khoảng lý tưởng: {round(f1_min, 2)} - {round(f1_max, 2)})")
print(f"2. Chỉ số rủi ro bất bình đẳng (Gini): {round(final_f2, 2)} (Khoảng lý tưởng: {round(f2_min, 2)} - {round(f2_max, 2)})")
print(f"3. Chỉ số lan tỏa AI toàn quốc: {round(final_f3, 2)} (Khoảng lý tưởng: {round(f3_min, 2)} - {round(f3_max, 2)})")

# 6. Vẽ đồ thị trực quan hóa dòng vốn đầu tư công nghệ giữa các vùng
plt.figure(figsize=(12, 6))
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

bars = plt.barh(df_res['Vùng Kinh Tế'], df_res['Ngân Sách Số Phân Bổ (Tỷ VND)'], color='#2ca02c', edgecolor='black', height=0.6)

# Thêm số liệu hiển thị trên từng cột
for bar in bars:
    width = bar.get_width()
    plt.text(width + 150, bar.get_y() + bar.get_height()/2, f'{int(width):,} tỷ', 
             va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')

plt.title('Kế Hoạch Phân Bổ Ngân Sách Số Quốc Gia Đa Mục Tiêu Đến 2025', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Mức Ngân Sách Đầu Tư Công Nghệ (Tỷ VND)', fontsize=12)
plt.xlim(0, max(df_res['Ngân Sách Số Phân Bổ (Tỷ VND)']) + 1500)
plt.tight_layout()
plt.show()