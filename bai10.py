import pyomo.environ as pyo

# ==========================================
# KHỞI TẠO DỮ LIỆU TỪ BẢNG 10.3 & 10.4
# ==========================================
# Xác suất các kịch bản
p_dict = {'s1': 0.30, 's2': 0.45, 's3': 0.20, 's4': 0.05}

# Hệ số beta giai đoạn 1
beta_dict = {'I': 1.00, 'D': 1.10, 'AI': 1.25, 'H': 0.95}

# Hệ số beta_s giai đoạn 2 theo từng kịch bản
beta_s_dict = {
    ('s1', 'I'): 1.25, ('s1', 'D'): 1.35, ('s1', 'AI'): 1.55, ('s1', 'H'): 1.05,
    ('s2', 'I'): 1.00, ('s2', 'D'): 1.10, ('s2', 'AI'): 1.25, ('s2', 'H'): 0.95,
    ('s3', 'I'): 0.75, ('s3', 'D'): 0.85, ('s3', 'AI'): 0.90, ('s3', 'H'): 1.00,
    ('s4', 'I'): 0.40, ('s4', 'D'): 0.50, ('s4', 'AI'): 0.55, ('s4', 'H'): 1.10
}

solver = pyo.SolverFactory('appsi_highs')

# ==========================================
# HÀM XÂY DỰNG MÔ HÌNH NGẪU NHIÊN (SP - Stochastic Problem)
# ==========================================
def create_sp_model():
    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=['I', 'D', 'AI', 'H'])
    m.S = pyo.Set(initialize=['s1', 's2', 's3', 's4'])
    
    m.p = pyo.Param(m.S, initialize=p_dict)
    m.beta = pyo.Param(m.J, initialize=beta_dict)
    m.beta_s = pyo.Param(m.S, m.J, initialize=beta_s_dict)
    
    # Biến quyết định
    m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)       # First-stage
    m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)  # Second-stage
    
    # Ràng buộc ngân sách giai đoạn 1 (<= 65.000)
    m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= 65000)
    
    # Ràng buộc ngân sách dự phòng giai đoạn 2 (<= 15.000)
    def budget2_rule(m, s):
        return sum(m.y[s, j] for j in m.J) <= 15000
    m.budget2 = pyo.Constraint(m.S, rule=budget2_rule)
    
    # Ràng buộc nhân lực AI: y_AI^s <= 0.5 * x_H
    def ai_constraint_rule(m, s):
        return m.y[s, 'AI'] <= 0.5 * m.x['H']
    m.ai_constraint = pyo.Constraint(m.S, rule=ai_constraint_rule)
    
    # Hàm mục tiêu: Tối đa hóa GDP kỳ vọng
    def obj_rule(m):
        first = sum(m.beta[j] * m.x[j] for j in m.J)
        second = sum(m.p[s] * sum(m.beta_s[s, j] * m.y[s, j] for j in m.J) for s in m.S)
        return first + second
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    
    return m

# ==========================================
# CÂU 10.5.1: GIẢI BÀI TOÁN STOCHASTIC (SP)
# ==========================================
sp_model = create_sp_model()
solver.solve(sp_model)
SP_val = pyo.value(sp_model.obj)

print("--- 10.5.1 KẾT QUẢ MÔ HÌNH NGẪU NHIÊN (SP) ---")
print("Quyết định First-stage (Here-and-now):")
for j in sp_model.J:
    print(f"  x[{j}] = {pyo.value(sp_model.x[j]):.2f} tỷ VND")
print(f"Giá trị GDP kỳ vọng (SP): {SP_val:.2f}\n")

# ==========================================
# CÂU 10.5.2 & 10.5.3: TÍNH VSS (Value of Stochastic Solution)
# ==========================================
# Bước 1: Tính Beta_s kỳ vọng
exp_beta_s = {j: sum(p_dict[s] * beta_s_dict[s, j] for s in p_dict) for j in beta_dict}

# Bước 2: Lập mô hình EV (Expected Value Problem)
ev_model = pyo.ConcreteModel()
ev_model.J = pyo.Set(initialize=['I', 'D', 'AI', 'H'])
ev_model.x = pyo.Var(ev_model.J, within=pyo.NonNegativeReals)
ev_model.y = pyo.Var(ev_model.J, within=pyo.NonNegativeReals)

ev_model.b1 = pyo.Constraint(expr=sum(ev_model.x[j] for j in ev_model.J) <= 65000)
ev_model.b2 = pyo.Constraint(expr=sum(ev_model.y[j] for j in ev_model.J) <= 15000)
ev_model.ai_c = pyo.Constraint(expr=ev_model.y['AI'] <= 0.5 * ev_model.x['H'])

ev_model.obj = pyo.Objective(
    expr=sum(beta_dict[j] * ev_model.x[j] for j in ev_model.J) + 
         sum(exp_beta_s[j] * ev_model.y[j] for j in ev_model.J), 
    sense=pyo.maximize
)
solver.solve(ev_model)

# Lưu lại nghiệm x_EV
x_EV = {j: pyo.value(ev_model.x[j]) for j in ev_model.J}

# Bước 3: Đưa x_EV vào mô hình SP để tính EEV (Expected result of Expected Value)
eev_model = create_sp_model()
for j in eev_model.J:
    eev_model.x[j].fix(x_EV[j]) # Đóng băng quyết định x bằng x_EV
solver.solve(eev_model)
EEV_val = pyo.value(eev_model.obj)
VSS = SP_val - EEV_val

print("--- 10.5.2 & 10.5.3: TÍNH VSS ---")
print("Quyết định First-stage nếu dùng mô hình trung bình (EV):")
for j in ev_model.J:
    print(f"  x_EV[{j}] = {x_EV[j]:.2f} tỷ VND")
print(f"Giá trị EEV: {EEV_val:.2f}")
print(f"Giá trị VSS (SP - EEV): {VSS:.2f} tỷ VND")
print("-> Ý nghĩa: VSS đo lường rủi ro (hoặc giá trị mất đi) nếu Chính phủ chỉ dựa vào 1 kịch bản trung bình thay vì tư duy xác suất.\n")

# ==========================================
# CÂU 10.5.3: TÍNH EVPI (Expected Value of Perfect Information)
# ==========================================
# Giải bài toán "Wait-and-See" (Biết trước kịch bản nào sẽ xảy ra ngay từ đầu)
WS_val = 0
for s in p_dict.keys():
    ws_model = pyo.ConcreteModel()
    ws_model.J = pyo.Set(initialize=['I', 'D', 'AI', 'H'])
    ws_model.x = pyo.Var(ws_model.J, within=pyo.NonNegativeReals)
    ws_model.y = pyo.Var(ws_model.J, within=pyo.NonNegativeReals)
    
    ws_model.b1 = pyo.Constraint(expr=sum(ws_model.x[j] for j in ws_model.J) <= 65000)
    ws_model.b2 = pyo.Constraint(expr=sum(ws_model.y[j] for j in ws_model.J) <= 15000)
    ws_model.ai_c = pyo.Constraint(expr=ws_model.y['AI'] <= 0.5 * ws_model.x['H'])
    
    ws_model.obj = pyo.Objective(
        expr=sum(beta_dict[j] * ws_model.x[j] for j in ws_model.J) + 
             sum(beta_s_dict[s, j] * ws_model.y[j] for j in ws_model.J), 
        sense=pyo.maximize
    )
    solver.solve(ws_model)
    WS_val += p_dict[s] * pyo.value(ws_model.obj)

EVPI = WS_val - SP_val
print("--- 10.5.3: TÍNH EVPI ---")
print(f"Giá trị Wait-and-See (WS): {WS_val:.2f}")
print(f"Giá trị EVPI (WS - SP): {EVPI:.2f} tỷ VND")
print("-> Ý nghĩa: Đây là số tiền tối đa Chính phủ sẵn sàng chi trả để dự báo chính xác 100% nền kinh tế thế giới.")