import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# KHỞI TẠO DỮ LIỆU TỪ BẢNG 9.3
# ==========================================
sectors = [
    "Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Bán buôn bán lẻ", 
    "Tài chính-Ngân hàng", "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo"
]
N = len(sectors)

# Dữ liệu lao động (quy đổi từ triệu người ra số người thực tế)
L_million = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
L_jobs = L_million * 1_000_000

# Các tham số rủi ro và hệ số tác động (việc/tỷ)
risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
b1 = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0])
c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
d1 = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0])

# ==========================================
# CÂU 9.4.1: CÀI ĐẶT MÔ HÌNH VÀ GIẢI BÀI TOÁN
# ==========================================
print("--- KẾT QUẢ CÂU 9.4.1: PHÂN BỔ TỐI ƯU ---")
x_AI = cp.Variable(N, nonneg=True)
x_H = cp.Variable(N, nonneg=True)

# Các phương trình động lực học việc làm
NewJob = cp.multiply(a1, x_AI)
Upgrade = cp.multiply(b1, x_H)
Displaced = cp.multiply(cp.multiply(c1, risk), x_AI)
RetrainCap = cp.multiply(d1, x_H)
NetJob = NewJob + Upgrade - Displaced

# Ràng buộc cơ bản
constraints = [
    cp.sum(x_AI + x_H) <= 30000,
    NetJob >= 0,
    Displaced <= RetrainCap
]

# Hàm mục tiêu: Tối đa hóa tổng NetJob
prob = cp.Problem(cp.Maximize(cp.sum(NetJob)), constraints)
prob.solve()

# In kết quả
results_df = pd.DataFrame({
    'Ngành': sectors,
    'x_AI (tỷ VND)': np.round(x_AI.value, 2),
    'x_H (tỷ VND)': np.round(x_H.value, 2),
    'NetJob ròng': np.round(NetJob.value, 0)
})
print(results_df.to_string(index=False))
print(f"\nTổng NetJob ròng hệ thống: {np.sum(NetJob.value):,.0f} việc làm\n")

# ==========================================
# CÂU 9.4.2: NGƯỠNG ĐẦU TƯ ĐÀO TẠO TẠI NGÀNH 2 (CN CHẾ BIẾN CHẾ TẠO)
# ==========================================
print("--- KẾT QUẢ CÂU 9.4.2: PHÂN TÍCH NGƯỠNG NGÀNH 2 ---")
# Ngành 2 có index = 1
idx = 1 
# Giả sử ngân sách AI đổ dồn tối đa vào ngành 2
max_x_AI_2 = 30000 
displaced_2 = c1[idx] * risk[idx] * max_x_AI_2

# Ràng buộc 1: RetrainCap >= Displaced => d1 * x_H >= displaced_2
min_x_H_retrain = displaced_2 / d1[idx]

# Ràng buộc 2: NetJob >= 0 => a1*x_AI + b1*x_H - Displaced >= 0
min_x_H_netjob = (displaced_2 - a1[idx] * max_x_AI_2) / b1[idx]

# Chọn ngưỡng lớn nhất để thỏa mãn cả 2 điều kiện
threshold_x_H = max(min_x_H_retrain, min_x_H_netjob)

print(f"Để chịu được cú sốc khi đầu tư {max_x_AI_2} tỷ vào AI cho ngành CN Chế biến chế tạo:")
print(f"-> Cần tối thiểu {threshold_x_H:,.2f} tỷ VND cho đào tạo lại (x_H).")
print(f"Giải thích: Hệ số c1 (62.4) và Risk (42%) của ngành này rất cao, tạo ra rủi ro đào thải lớn.\n")

# ==========================================
# CÂU 9.4.3: MÔ PHỎNG SANKEY DIAGRAM (LAO ĐỘNG PHỔ THÔNG NGÀNH 1, 3, 4)
# ==========================================
# Index của ngành 1, 3, 4 trong mảng là 0, 2, 3
target_indices = [0, 2, 3]

# Tính toán lại giá trị thực tế sau khi đã tối ưu ở 9.4.1
displaced_vals = Displaced.value[target_indices]
upgrade_vals = Upgrade.value[target_indices]
# Giả định phần lao động còn lại không bị ảnh hưởng
retained_vals = L_jobs[target_indices] - displaced_vals

source_labels = [sectors[i] for i in target_indices]
target_labels = ["Bị thay thế (Displaced)", "Nâng cấp (Upgraded)", "Giữ nguyên (Retained)"]

# Xây dựng luồng dữ liệu cho Sankey
sources = [0, 0, 0, 1, 1, 1, 2, 2, 2] # Các ngành nguồn
targets = [3, 4, 5, 3, 4, 5, 3, 4, 5] # Các trạng thái đích
values = [
    displaced_vals[0], upgrade_vals[0], retained_vals[0],
    displaced_vals[1], upgrade_vals[1], retained_vals[1],
    displaced_vals[2], upgrade_vals[2], retained_vals[2]
]

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15, thickness=20,
        line=dict(color="black", width=0.5),
        label=source_labels + target_labels
    ),
    link=dict(source=sources, target=targets, value=values)
)])
fig.update_layout(title_text="Luồng dịch chuyển lao động (Ngành 1, 3, 4)", font_size=10)
# Uncomment dòng dưới để hiển thị biểu đồ trên trình duyệt
# fig.show()

# ==========================================
# CÂU 9.4.4: MỞ RỘNG - RÀNG BUỘC KHÔNG QUÁ 5% LAO ĐỘNG
# ==========================================
print("--- KẾT QUẢ CÂU 9.4.4: KIỂM TRA TÍNH KHẢ THI VỚI RÀNG BUỘC MỚI ---")
constraints_extended = constraints.copy()
# Ràng buộc: Displaced <= 0.05 * L_jobs
constraints_extended.append(Displaced <= 0.05 * L_jobs)

prob_extended = cp.Problem(cp.Maximize(cp.sum(NetJob)), constraints_extended)
prob_extended.solve()

if prob_extended.status == cp.OPTIMAL:
    print(f"Bài toán VẪN KHẢ THI. Tổng NetJob mới: {np.sum(NetJob.value):,.0f}")
else:
    print("Bài toán KHÔNG KHẢ THI (Infeasible).")
    print("Giải thích: Ràng buộc an sinh 5% quá ngặt nghèo, khiến hệ thống không thể giải ngân tối đa ngân sách AI nhằm tối ưu việc làm mới.")