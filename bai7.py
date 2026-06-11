import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# ==========================================
# CÂU 7.4.1: ĐỊNH NGHĨA BÀI TOÁN VÀ CHẠY NSGA-II
# ==========================================

class VietnamDigitalProblem(ElementwiseProblem):
    def __init__(self):
        # 24 biến, 4 mục tiêu, 12 ràng buộc bất đẳng thức (<= 0)
        super().__init__(n_var=24, n_obj=4, n_ieq_constr=12,
                         xl=np.zeros(24), xu=np.ones(24)*12000)
        
        # Ma trận beta (6 vùng x 4 hạng mục) - Giả định số liệu từ Bài 4
        # Các hạng mục: I (Hạ tầng), D (Dữ liệu), AI (Thông minh), H (Nhân lực)
        self.beta = np.array([
            [0.25, 0.30, 0.35, 0.20],  # Vùng 1
            [0.22, 0.28, 0.33, 0.25],  # Vùng 2
            [0.20, 0.25, 0.30, 0.22],  # Vùng 3
            [0.18, 0.22, 0.28, 0.20],  # Vùng 4
            [0.26, 0.32, 0.38, 0.24],  # Vùng 5
            [0.15, 0.20, 0.25, 0.18]   # Vùng 6
        ])
        
        # Các hệ số môi trường và rủi ro
        self.e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
        self.rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
        self.sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

    def _evaluate(self, x, out, *args, **kwargs):
        X = x.reshape(6, 4) # Định dạng lại thành ma trận (6 vùng, 4 hạng mục)
        
        # --- CÁC HÀM MỤC TIÊU (Pymoo mặc định là MINIMIZE) ---
        # f1: Max GDP gain => Đảo dấu thành Minimize
        f1 = -(self.beta * X).sum()
        
        # f2: Gini xấp xỉ bằng MAD chuẩn hóa của tổng đầu tư các vùng
        sums_vung = X.sum(axis=1)
        f2 = np.abs(sums_vung - sums_vung.mean()).mean()
        
        # f3: Phát thải (Chỉ tính trên Hạ tầng X[:,0] và AI X[:,2])
        f3 = (self.e * (X[:,0] + X[:,2])).sum()
        
        # f4: Rủi ro ròng
        f4 = (self.rho * X[:,2]).sum() - (self.sig * X[:,3]).sum()
        
        out['F'] = [f1, f2, f3, f4]
        
        # --- CÁC RÀNG BUỘC CHÍNH SÁCH (Dạng g(x) <= 0) ---
        g = np.zeros(12)
        
        # g[0]: Tổng ngân sách toàn quốc <= 35,000
        g[0] = X.sum() - 35000
        
        # g[1:7]: Ngân sách tối đa của mỗi vùng trong 6 vùng <= 12,000
        for i in range(6):
            g[1 + i] = X[i, :].sum() - 12000
            
        # g[7:11]: Ngân sách tối thiểu cho từng danh mục đầu tư trên cả nước >= 4,000
        # Chuyển vế thành: 4000 - Tổng_hạng_mục <= 0
        sums_hang_muc = X.sum(axis=0)
        for j in range(4):
            g[7 + j] = 4000 - sums_hang_muc[j]
            
        # g[11]: Phát thải môi trường không vượt ngưỡng an toàn (ví dụ: 15,000)
        g[11] = f3 - 15000
        
        out['G'] = g

# Thực thi tối ưu hóa bằng NSGA-II
problem = VietnamDigitalProblem()
algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=42,
               verbose=False)

# ==========================================
# CÂU 7.4.2: TRÍCH XUẤT VÀ TRỰC QUAN HÓA PARETO
# ==========================================

# Trích xuất tập Pareto (Các nghiệm không bị trội)
X_pareto = res.X
F_pareto = res.F

# Lưu ý: Chuyển f1 về lại giá trị dương để vẽ đồ thị đúng ý nghĩa "Tăng trưởng GDP"
F_plot = F_pareto.copy()
F_plot[:, 0] = -F_plot[:, 0] 

print(f"Số lượng nghiệm tối ưu Pareto tìm được: {len(F_pareto)}")

# 1. Vẽ biểu đồ Scatter 3D (f1, f2, f3)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(F_plot[:, 0], F_plot[:, 1], F_plot[:, 2], c=F_plot[:, 3], cmap='viridis', s=40)
ax.set_xlabel('f1: GDP Gain (Max)')
ax.set_ylabel('f2: Bất bình đẳng MAD (Min)')
ax.set_zlabel('f3: Phát thải (Min)')
plt.title('Biểu đồ Scatter 3D của tập Pareto (Màu sắc thể hiện f4: Rủi ro)')
fig.colorbar(sc, label='f4: Rủi ro ròng (Min)')
plt.tight_layout()
plt.show()

# 2. Vẽ biểu đồ song song (Parallel Coordinates) cho cả 4 mục tiêu
fig, ax = plt.subplots(figsize=(10, 5))
labels = ['f1: GDP Gain\n(Thang lớn)', 'f2: Bất bình đẳng\n(Thang nhỏ)', 'f3: Phát thải\n(Thang lớn)', 'f4: Rủi ro ròng\n(Thang trung bình)']

# Chuẩn hóa min-max để trực quan hóa trên cùng một hệ trục không bị lệch thang đo
F_norm = (F_plot - F_plot.min(axis=0)) / (F_plot.max(axis=0) - F_plot.min(axis=0))

for i in range(len(F_norm)):
    ax.plot(labels, F_norm[i, :], color='teal', alpha=0.3)
ax.set_title("Biểu đồ song song (Parallel Coordinates) - Đã chuẩn hóa dữ liệu")
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# CÂU 7.4.3: ÁP DỤNG TOPSIS CHỌN NGHIỆM THỎA HIỆP
# ==========================================

# Trọng số ưu tiên chính sách: f1 (0.40), f2 (0.25), f3 (0.20), f4 (0.15)
weights = np.array([0.40, 0.25, 0.20, 0.15])

# Chuẩn hóa ma trận quyết định (sử dụng F_pareto gốc vì TOPSIS xử lý dạng tìm giá trị nhỏ nhất)
# Tất cả f1, f2, f3, f4 trong F_pareto hiện tại đều đang ở dạng CẦN GIẢM THIỂU (Minimize)
norm_matrix = F_pareto / np.sqrt((F_pareto**2).sum(axis=0))

# Nhân trọng số
weighted_matrix = norm_matrix * weights

# Xác định nghiệm lý tưởng (Ý tưởng tốt nhất là Min, tệ nhất là Max đối với các hàm mục tiêu gốc đã chuyển đổi)
ideal_best = weighted_matrix.min(axis=0)
ideal_worst = weighted_matrix.max(axis=0)

# Tính khoảng cách Euclidean đến nghiệm lý tưởng và lý tưởng âm
d_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
d_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))

# Tính điểm tương tự TOPSIS (Càng gần 1 càng tốt)
topsis_scores = d_worst / (d_best + d_worst)

# Nghiệm thỏa hiệp là nghiệm có điểm TOPSIS cao nhất
best_idx = np.argmax(topsis_scores)
compromise_solution_F = F_plot[best_idx]

print("\n--- NGHIỆM THỎA HIỆP ĐƯỢC CHỌN BỞI TOPSIS ---")
print(f"Chỉ số nghiệm trong tập Pareto: {best_idx}")
print(f"f1 (GDP Gain - Cực đại):  {compromise_solution_F[0]:.2f}")
print(f"f2 (Bất bình đẳng - Cực tiểu): {compromise_solution_F[1]:.2f}")
print(f"f3 (Phát thải - Cực tiểu):     {compromise_solution_F[2]:.2f}")
print(f"f4 (Rủi ro ròng - Cực tiểu):   {compromise_solution_F[3]:.2f}")

# ==========================================
# CÂU 7.4.4: PHÂN TÍCH CHI PHÍ CƠ HỘI
# ==========================================

# Tìm nghiệm có Tăng trưởng GDP (f1 dương) cao nhất trong tập Pareto
max_gdp_idx = np.argmax(F_plot[:, 0])
max_gdp_solution_F = F_plot[max_gdp_idx]

# Tính toán mức độ hi sinh (%) của nghiệm Tăng trưởng cao nhất so với nghiệm Thỏa hiệp
# Công thức: ((Giá trị_MaxGDP - Giá trị_ThỏaHiệp) / Giá trị_ThỏaHiệp) * 100
sacrifice_f2 = ((max_gdp_solution_F[1] - compromise_solution_F[1]) / compromise_solution_F[1]) * 100
sacrifice_f3 = ((max_gdp_solution_F[2] - compromise_solution_F[2]) / compromise_solution_F[2]) * 100
gdp_gain_increase = ((max_gdp_solution_F[0] - compromise_solution_F[0]) / compromise_solution_F[0]) * 100

print("\n--- PHÂN TÍCH CHI PHÍ CƠ HỘI ---")
print(f"Nghiệm có GDP cao nhất đạt được f1 = {max_gdp_solution_F[0]:.2f} (Tăng {gdp_gain_increase:.2f}% so với thỏa hiệp).")
print(f"Để đạt mức tăng trưởng cực đại này, chính sách phải đánh đổi:")
print(f"  - Tăng mức bất bình đẳng (f2) thêm: {sacrifice_f2:.2f}% (Tệ hơn)")
print(f"  - Tăng lượng phát thải (f3) thêm:    {sacrifice_f3:.2f}% (Tệ hơn)")