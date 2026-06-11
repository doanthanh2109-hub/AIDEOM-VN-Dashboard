import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# THAM SỐ MÔ HÌNH (Theo Chiến lược Quốc gia & Dữ liệu thực tế)
# ---------------------------------------------------------
T = 10  # Giai đoạn 10 năm: 2026 - 2035
rho = 0.97  # Hệ số chiết khấu thời gian

# Hệ số co giãn (đóng góp biên vào GDP)
alpha_K = 0.33   # Vốn vật chất
alpha_D = 0.10   # Hạ tầng số
alpha_AI = 0.08  # Công nghệ AI
alpha_H = 0.07   # Nhân lực số
alpha_L = 1 - (alpha_K + alpha_D + alpha_AI + alpha_H) # Lao động phổ thông (0.42)

# Tỷ lệ khấu hao
delta_K = 0.05
delta_D = 0.12
delta_AI = 0.15
delta_H = 0.02  # Tốc độ mai một kỹ năng nếu không tái đào tạo

# Lực lượng lao động ổn định (triệu người)
L_val = 54.0
A_val = 1.0  # Năng suất tổng nhân tố (TFP) cơ sở

# Giá trị ban đầu đầu kỳ (Năm 2026)
K0 = 27500.0  # Nghìn tỷ VND
D0 = 20.3     # Chỉ số hạ tầng kết nối số
AI0 = 86.0    # Điểm sẵn sàng AI
H0 = 30.0     # % Lao động qua đào tạo số

# ---------------------------------------------------------
# KHAI BÁO BIẾN TRẠNG THÁI & BIẾN ĐIỀU KHIỂN (Xử lý không gian trạng thái)
# ---------------------------------------------------------
# Biến trạng thái cuối kỳ (T+1 để chứa giá trị tích lũy cho năm sau)
K = cp.Variable(T+1, nonneg=True)
D = cp.Variable(T+1, nonneg=True)
AI = cp.Variable(T+1, nonneg=True)
H = cp.Variable(T+1, nonneg=True)

# Biến điều khiển dòng vốn đầu tư từng năm
I_K = cp.Variable(T, nonneg=True)
I_D = cp.Variable(T, nonneg=True)
I_AI = cp.Variable(T, nonneg=True)
I_H = cp.Variable(T, nonneg=True)

# Biến phụ trợ Log để chuyển đổi bài toán sang dạng Quy hoạch hình học lồi (GP)
log_K = cp.Variable(T, nonneg=False)
log_D = cp.Variable(T, nonneg=False)
log_AI = cp.Variable(T, nonneg=False)
log_H = cp.Variable(T, nonneg=False)
log_Y = cp.Variable(T, nonneg=False)

# Biến tiêu dùng
C = cp.Variable(T, nonneg=True)

# ---------------------------------------------------------
# THIẾT LẬP HỆ THỐNG RÀNG BUỘC ĐỘNG
# ---------------------------------------------------------
constraints = [
    K[0] == K0,
    D[0] == D0,
    AI[0] == AI0,
    H[0] == H0
]

for t in range(T):
    # Cấu trúc liên kết biến log để duy trì tính lồi
    constraints += [
        log_K[t] <= cp.log(K[t]),
        log_D[t] <= cp.log(D[t]),
        log_AI[t] <= cp.log(AI[t]),
        log_H[t] <= cp.log(H[t])
    ]
    
    # Định nghĩa hàm sản xuất Cobb-Douglas dạng tuyến tính hóa logarit
    constraints += [
        log_Y[t] == np.log(A_val) + alpha_K * log_K[t] + alpha_D * log_D[t] + \
                    alpha_AI * log_AI[t] + alpha_H * log_H[t] + alpha_L * np.log(L_val)
    ]
    
    # Ràng buộc phân bổ nguồn lực (Sản lượng >= Tiêu dùng + Tổng đầu tư các hạng mục)
    constraints += [
        C[t] + I_K[t] + I_D[t] + I_AI[t] + I_H[t] <= cp.exp(log_Y[t])
    ]
    
    # Phương trình chuyển dịch trạng thái của các nguồn vốn sản xuất
    constraints += [
        K[t+1] == (1 - delta_K) * K[t] + I_K[t],
        D[t+1] == (1 - delta_D) * D[t] + I_D[t],
        AI[t+1] == (1 - delta_AI) * AI[t] + I_AI[t],
        H[t+1] == (1 - delta_H) * H[t] + 0.8 * I_H[t]  # Hiệu suất đào tạo công nghệ 0.8
    ]

# ---------------------------------------------------------
# HÀM MỤC TIÊU & GIẢI MÔ HÌNH
# ---------------------------------------------------------
# Tối đa hóa tổng hữu dụng phúc lợi xã hội chiết khấu theo thời gian
objective = cp.Maximize(sum(rho**t * cp.log(C[t]) for t in range(T)))

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS, verbose=False)

# ---------------------------------------------------------
# TRÍCH XUẤT KẾT QUẢ & TRỰC QUAN HÓA QUỸ ĐẠO TĂNG TRƯỞNG
# ---------------------------------------------------------
if prob.status in ["optimal", "optimal_inaccurate"]:
    years = np.arange(2026, 2026 + T)
    
    C_val = C.value
    K_val = K.value[:-1]
    D_val = D.value[:-1]
    AI_val = AI.value[:-1]
    H_val = H.value[:-1]
    Y_val = np.exp(log_Y.value)
    
    print("=== KẾT QUẢ TỐI ƯU HÓA QUỸ ĐẠO ĐỘNG RAMSEY (AIDEOM-VN) ===")
    print(f"Tổng giá trị hàm hữu dụng (Phúc lợi xã hội tích lũy): {prob.value:.4f}")
    print("-" * 75)
    print(f"{'Năm':<6}{'Sản lượng Y':<15}{'Tiêu dùng C':<15}{'Vốn số D':<12}{'Điểm AI':<12}{'Nhân lực H':<12}")
    print("-" * 75)
    for i, yr in enumerate(years):
        print(f"{yr:<6}{Y_val[i]:<15.2f}{C_val[i]:<15.2f}{D_val[i]:<12.2f}{AI_val[i]:<12.2f}{H_val[i]:<12.2f}")
    print("-" * 75)
    
    # Tiến hành vẽ đồ thị phân tích động thái kinh tế vĩ mô
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Đồ thị 1: Quỹ đạo Sản lượng và Tiêu dùng qua các năm
    axs[0, 0].plot(years, Y_val, marker='o', color='navy', label='Sản lượng GDP (Y)')
    axs[0, 0].plot(years, C_val, marker='s', color='darkorange', linestyle='--', label='Tiêu dùng công chúng (C)')
    axs[0, 0].set_title('Quỹ đạo Kinh tế vĩ mô: Sản lượng vs Tiêu dùng')
    axs[0, 0].set_xlabel('Năm')
    axs[0, 0].set_ylabel('Nghìn tỷ VND')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Đồ thị 2: Động thái tích lũy vốn vật chất cố định
    axs[0, 1].plot(years, K_val, marker='^', color='green', label='Vốn truyền thống (K)')
    axs[0, 1].set_title('Quỹ đạo Tích lũy Vốn vật chất truyền thống')
    axs[0, 1].set_xlabel('Năm')
    axs[0, 1].set_ylabel('Nghìn tỷ VND')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Đồ thị 3: Quỹ đạo phát triển tài sản vô hình công nghệ (D và AI)
    ax3_twin = axs[1, 0].twinx()
    p1 = axs[1, 0].plot(years, D_val, marker='v', color='purple', label='Hạ tầng số (D) [Trục trái]')
    p2 = ax3_twin.plot(years, AI_val, marker='d', color='crimson', linestyle='-.', label='Điểm Sẵn sàng AI (AI) [Trục phải]')
    axs[1, 0].set_title('Quỹ đạo Đột phá Hạ tầng Số và Trí tuệ nhân tạo')
    axs[1, 0].set_xlabel('Năm')
    axs[1, 0].set_ylabel('Chỉ số hạ tầng kết nối số')
    ax3_twin.set_ylabel('Thang điểm AI Readiness')
    axs[1, 0].grid(True)
    plots = p1 + p2
    labels = [l.get_label() for l in plots]
    axs[1, 0].legend(plots, labels, loc='upper left')
    
    # Đồ thị 4: Phát triển chất lượng Nhân lực công nghệ số
    axs[1, 1].plot(years, H_val, marker='h', color='teal', label='Tỷ lệ lao động số kỹ năng cao (H)')
    axs[1, 1].set_title('Quỹ đạo Nâng chất Nguồn nhân lực số Quốc gia')
    axs[1, 1].set_xlabel('Năm')
    axs[1, 1].set_ylabel('% Tỷ trọng trong lực lượng lao động')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
else:
    print("Mô hình không tìm thấy nghiệm tối ưu. Kiểm tra lại các điều kiện biên hoặc phương pháp đổi biến phụ.")