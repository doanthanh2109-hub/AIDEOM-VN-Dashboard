import pandas as pd
import pulp as pl
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc và chuẩn bị dữ liệu từ file csv ngành
try:
    df_sectors = pd.read_csv("vietnam_sectors_2024.csv")
    print("Đọc dữ liệu file vietnam_sectors_2024.csv thành công!")
except FileNotFoundError:
    print("Vui lòng đảm bảo file csv nằm cùng thư mục làm việc.")
    raise

# Danh sách tên các ngành kinh tế
sectors = df_sectors['sector_name_vi'].tolist()
N = len(sectors)

# 2. Khởi tạo các tham số cho Mô hình Ngẫu nhiên Hai Giai đoạn
B = 10000  # Tổng ngân sách giai đoạn 1 (tỷ VND)
x_min = 500  # Mức đầu tư tối thiểu cho mỗi ngành (tỷ VND)
x_max = 3000  # Mức đầu tư tối đa cho mỗi ngành (tỷ VND)

# Hệ số chi phí giai đoạn 1 (giả định đồng đều c_i = 1 để tối ưu phân bổ hiệu quả)
c = {s_name: 1.0 for s_name in sectors}

# Hệ số hiệu quả can thiệp ban đầu k_i (giảm số người thất nghiệp trên mỗi tỷ VND đầu tư)
# Ngành có chỉ số AI readiness cao hoặc rủi ro tự động hóa cao sẽ có phản ứng khác nhau
k = {}
for idx, row in df_sectors.iterrows():
    # Hệ số chuyển đổi giả định: đầu tư giúp giảm thiểu nguy cơ dựa trên mức độ sẵn sàng AI
    k[row['sector_name_vi']] = 0.0005 * (row['ai_readiness_0_100'] / 50.0)

# Định nghĩa các kịch bản giai đoạn 2
scenarios = ['AI_Baseline', 'AI_Surge']
probabilities = {'AI_Baseline': 0.70, 'AI_Surge': 0.30}
shock_factors = {'AI_Baseline': 1.0, 'AI_Surge': 1.8} # Kịch bản Surge tăng gấp 1.8 lần áp lực lao động

# Chi phí giai đoạn 2 (triệu VND trên 1 lao động)
# q: Chi phí đào tạo lại chủ động; d: Chi phí phạt do thất nghiệp tiêu cực (d > q)
q = {'AI_Baseline': 15, 'AI_Surge': 25}
d = {'AI_Baseline': 35, 'AI_Surge': 65}

# Tính toán lượng lao động có nguy cơ bị ảnh hưởng (Risk Labor) cho từng kịch bản (đơn vị: triệu người)
risk_labor = {}
for s_name in scenarios:
    risk_labor[s_name] = {}
    for idx, row in df_sectors.iterrows():
        sector = row['sector_name_vi']
        L_i = row['labor_million']
        auto_risk = row['automation_risk_pct'] / 100.0
        shock = shock_factors[s_name]
        risk_labor[s_name][sector] = L_i * auto_risk * shock

# 3. Khai báo bài toán tối ưu với PuLP
model = pl.LpProblem("Two_Stage_Stochastic_Budget_Allocation", pl.LpMinimize)

# Biến quyết định Giai đoạn 1: Ngân sách đầu tư ban đầu x_i
x = pl.LpVariable.dicts("X_Investment", sectors, lowBound=x_min, upBound=x_max, cat='Continuous')

# Biến quyết định Giai đoạn 2: 
# y_is: Lao động được đào tạo lại
# u_is: Lao động thất nghiệp không được giải quyết (Biến phụ để tuyến tính hóa)
y = pl.LpVariable.dicts("Y_Retrained", (sectors, scenarios), lowBound=0, cat='Continuous')
u = pl.LpVariable.dicts("U_Unresolved", (sectors, scenarios), lowBound=0, cat='Continuous')

# 4. Thiết lập Hàm mục tiêu (Objective Function)
# Min = Tổng chi phí đầu tư GĐ1 + Kỳ vọng (Chi phí đào tạo GĐ2 + Chi phí phạt thất nghiệp GĐ2)
obj_giai_doan_1 = pl.lpSum([c[i] * x[i] for i in sectors])
obj_giai_doan_2 = pl.lpSum([
    probabilities[s] * pl.lpSum([q[s] * y[i][s] + d[s] * u[i][s] for i in sectors])
    for s in scenarios
])
model += obj_giai_doan_1 + obj_giai_doan_2, "Total_Expected_Cost"

# 5. Thiết lập các Ràng buộc (Constraints)
# Ràng buộc Giai đoạn 1: Tổng ngân sách không vượt quá B
model += pl.lpSum([x[i] for i in sectors]) <= B, "Budget_Constraint_Stage1"

# Ràng buộc Giai đoạn 2 cho từng kịch bản và từng ngành
for s in scenarios:
    for i in sectors:
        # Ràng buộc cân bằng an sinh xã hội: u_is >= Risk_Labor_is - k_i * x_i - y_is
        model += u[i][s] >= risk_labor[s][i] - k[i] * x[i] - y[i][s], f"Social_Balance_{i}_{s}"

# 6. Tiến hành Giải mô hình
solver = pl.PULP_CBC_CMD(msg=False)
model.solve(solver)

print(f"Trạng thái tối ưu: {pl.LpStatus[model.status]}")

# 7. Thu thập kết quả và chuẩn bị dữ liệu báo cáo
results_gd1 = []
for i in sectors:
    results_gd1.append({
        'Ngành Kinh Tế': i,
        'Ngân Sách Đầu Tư GĐ1 (Tỷ VND)': round(x[i].varValue, 2)
    })
df_res_gd1 = pd.DataFrame(results_gd1)

results_gd2 = []
for s in scenarios:
    for i in sectors:
        results_gd2.append({
            'Kịch bản': s,
            'Ngành Kinh Tế': i,
            'Lao động nguy cơ (Triệu người)': round(risk_labor[s][i], 3),
            'Đào tạo lại khẩn cấp (Triệu người)': round(y[i][s].varValue, 3),
            'Thất nghiệp chưa giải quyết (Triệu người)': round(u[i][s].varValue, 3)
        })
df_res_gd2 = pd.DataFrame(results_gd2)

# Hiển thị bảng dữ liệu kết quả ra màn hình Terminal
print("\n--- KẾT QUẢ PHÂN BỔ NGÂN SÁCH GIAI ĐOẠN 1 ---")
print(df_res_gd1.to_string(index=False))

print("\n--- KẾT QUẢ ỨNG PHÓ AN SINH XÃ HỘI GIAI ĐOẠN 2 ---")
print(df_res_gd2.to_string(index=False))

# 8. Vẽ đồ thị trực quan hóa kết quả phân bổ chính sách kinh tế
sns.set_theme(style="whitegrid", font="DejaVu Sans") # Đảm bảo hiển thị tốt tiếng Việt nếu có font
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Đồ thị 1: Phân bổ ngân sách tối ưu ở giai đoạn 1
sns.barplot(
    data=df_res_gd1, 
    x='Ngân Sách Đầu Tư GĐ1 (Tỷ VND)', 
    y='Ngành Kinh Tế', 
    ax=axes[0], 
    palette='viridis'
)
axes[0].set_title('Phân bổ Ngân sách Đầu tư Phòng ngừa (Giai đoạn 1)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Tỷ VND')
axes[0].set_ylabel('')

# Đồ thị 2: So sánh số lao động thất nghiệp chưa được giải quyết giữa 2 kịch bản ở Giai đoạn 2
sns.barplot(
    data=df_res_gd2,
    x='Thất nghiệp chưa giải quyết (Triệu người)',
    y='Ngành Kinh Tế',
    hue='Kịch bản',
    ax=axes[1],
    palette='mako'
)
axes[1].set_title('Dự báo Thất nghiệp chưa giải quyết (Giai đoạn 2)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Triệu người')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()