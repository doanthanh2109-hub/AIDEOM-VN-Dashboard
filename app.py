import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CACHE DATA ĐỂ APP KHÔNG BỊ CHẬM ---
# Thêm @st.cache_data trước các hàm nặng để Streamlit nhớ kết quả, không chạy lại nhiều lần
@st.cache_data
def load_dummy_data():
    # Hàm này mô phỏng việc gọi dữ liệu từ M1 đến M5
    return pd.DataFrame({
        "Năm": [2026, 2027, 2028, 2029, 2030],
        "GDP (nghìn tỷ)": [12800, 13700, 14800, 16000, 17500]
    })

# --- THIẾT LẬP GIAO DIỆN CHUNG ---
st.set_page_config(page_title="AIDEOM-VN Dashboard", layout="wide")
st.title("Mô hình Ra quyết định AIDEOM-VN")
st.markdown("Hệ thống hỗ trợ phân bổ ngân sách số và đánh giá rủi ro.")

# --- KHU VỰC ĐIỀU KHIỂN (SIDEBAR) ---
st.sidebar.header("Tùy chỉnh Kịch bản")

# Dropdown chọn 5 kịch bản
scenario = st.sidebar.selectbox(
    "Chọn kịch bản chính sách:",
    ("S1. Truyền thống", "S2. Số hóa nhanh", "S3. AI dẫn dắt", "S4. Bao trùm số", "S5. Tối ưu cân bằng")
)

st.sidebar.divider()
st.sidebar.info("Đồ án cuối kỳ - Nhóm XYZ")

# --- KHU VỰC HIỂN THỊ CHÍNH (4 TABS) ---
# Tạo 4 tab theo đúng yêu cầu đề bài
tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan", "Phân bổ Ngân sách", "Kịch bản so sánh", "Cảnh báo rủi ro"])

with tab1:
    st.header("Tổng quan Kinh tế - Kịch bản: " + scenario)
    # Hiển thị số liệu nổi bật (KPI metrics)
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Dự báo GDP 2030", value="17,500 Tỷ", delta="+7.5%")
    col2.metric(label="Digital Index", value="65/100", delta="Tăng")
    col3.metric(label="Việc làm ròng (NetJob)", value="1.2 Triệu", delta="-50k")
    
    # Biểu đồ Plotly
    df_gdp = load_dummy_data()
    fig = px.line(df_gdp, x="Năm", y="GDP (nghìn tỷ)", title="Quỹ đạo Tăng trưởng GDP")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Phân bổ Ngân sách")
    st.write("Trực quan hóa ma trận phân bổ từ Module 3.")
    # Giả lập biểu đồ tròn
    df_budget = pd.DataFrame({"Hạng mục": ["K", "D", "AI", "H"], "Tỷ lệ": [70, 10, 10, 10]})
    fig_pie = px.pie(df_budget, values="Tỷ lệ", names="Hạng mục", title="Cơ cấu ngân sách")
    st.plotly_chart(fig_pie)

with tab3:
    st.header("Kịch bản So sánh")
    st.write("Bảng so sánh tổng hợp các chỉ số năm 2030 giữa S1, S3 và S5.")
    # Hiển thị bảng DataFrame
    st.dataframe(df_gdp)

with tab4:
    st.header("Cảnh báo Rủi ro")
    st.write("Cảnh báo an ninh mạng và dịch chuyển lao động từ Module 4 & 5.")
    st.warning("Cảnh báo: Ngành Chế biến chế tạo có rủi ro mất việc làm vượt ngưỡng 5%!")
    import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pulp

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & DATA (Tích hợp từ các bài trước)
# ==========================================
st.set_page_config(page_title="AIDEOM-VN Dashboard", layout="wide")

# --- DATA BÀI 1: MACRO & COBB-DOUGLAS ---
alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
K_2025, L_2025, D_2025, AI_2025, H_2025 = 25900, 53.4, 19.5, 80.1, 29.2
A_bar = 0.98  # TFP trung bình

# --- DATA BÀI 9: LAO ĐỘNG & RỦI RO ---
sectors = ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Bán buôn bán lẻ", 
           "Tài chính-Ngân hàng", "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo"]
L_jobs = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15]) * 1_000_000
risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
b1 = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0])
c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
d1 = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0])

# ==========================================
# 2. CÁC HÀM XỬ LÝ LÕI (MODULES)
# ==========================================

@st.cache_data
def run_m3_optimization(total_budget=100000):
    """M3: Giải bài toán LP Phân bổ tối ưu (Tích hợp từ Bài 4)"""
    regions = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
    items = ['K', 'D', 'AI', 'H']
    
    # Ma trận beta giả định tác động
    beta_matrix = np.array([
        [1.15, 0.85, 0.55, 1.30], [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15], [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00], [1.10, 0.85, 0.65, 1.25]
    ])
    
    prob = pulp.LpProblem("S5_Optimization", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", ((r, j) for r in range(6) for j in range(4)), lowBound=0)
    
    # Hàm mục tiêu
    prob += pulp.lpSum(beta_matrix[r, j] * x[r, j] for r in range(6) for j in range(4))
    
    # Ràng buộc ngân sách
    prob += pulp.lpSum(x[r, j] for r in range(6) for j in range(4)) <= total_budget
    for r in range(6):
        prob += pulp.lpSum(x[r, j] for j in range(4)) >= total_budget * 0.05 # Tối thiểu 5% mỗi vùng
        prob += pulp.lpSum(x[r, j] for j in range(4)) <= total_budget * 0.30 # Tối đa 30% mỗi vùng
        
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Gom nhóm kết quả thành tỷ lệ
    item_totals = [sum(x[r, j].varValue for r in range(6)) for j in range(4)]
    total_spent = sum(item_totals)
    ratios = [val / total_spent for val in item_totals] if total_spent > 0 else [0.25]*4
    return ratios

def run_m1_forecasting(ratios, total_budget=100000):
    """M1: Dự báo GDP bằng Cobb-Douglas (Tích hợp từ Bài 1)"""
    # Phân bổ ngân sách thực tế
    budget_alloc = [r * total_budget for r in ratios]
    
    # Cập nhật vốn tích lũy cho năm 2030 (Giả lập đơn giản hóa)
    years_forward = 5
    K_2030 = K_2025 + budget_alloc[0] * 0.5 
    L_2030 = L_2025 * (1.01 ** years_forward)
    D_2030 = D_2025 + budget_alloc[1] * 0.001
    AI_2030 = AI_2025 + budget_alloc[2] * 0.005
    H_2030 = H_2025 + budget_alloc[3] * 0.002
    
    A_2030 = A_bar * (1.012 ** years_forward)
    
    GDP_2030 = A_2030 * (K_2030**alpha) * (L_2030**beta) * (D_2030**gamma) * (AI_2030**delta) * (H_2030**theta)
    return GDP_2030, budget_alloc

def run_m4_labor(ai_budget, h_budget):
    """M4: Tính toán NetJob và luồng lao động (Tích hợp từ Bài 9)"""
    # Chia đều ngân sách cho 8 ngành (Trong thực tế M3 sẽ chia chi tiết hơn)
    x_AI = np.ones(8) * (ai_budget / 8)
    x_H = np.ones(8) * (h_budget / 8)
    
    NewJob = a1 * x_AI
    Upgrade = b1 * x_H
    Displaced = c1 * risk * x_AI
    NetJob = NewJob + Upgrade - Displaced
    return Displaced, Upgrade, NetJob

# ==========================================
# 3. LUỒNG ĐIỀU KHIỂN GIAO DIỆN (UI/UX)
# ==========================================

st.title("AIDEOM-VN: Hệ thống Hỗ trợ Ra quyết định")
st.markdown("Nguyên mẫu tích hợp 6 Module dự báo và tối ưu hóa chính sách.")

# --- SIDEBAR ---
st.sidebar.header("Tùy chỉnh Kịch bản")
scenario_dict = {
    "S1. Truyền thống": [0.70, 0.10, 0.10, 0.10],
    "S2. Số hóa nhanh": [0.25, 0.45, 0.15, 0.15],
    "S3. AI dẫn dắt": [0.20, 0.20, 0.45, 0.15],
    "S4. Bao trùm số": [0.30, 0.20, 0.10, 0.40],
    "S5. Tối ưu cân bằng": "OPTIMIZE"
}

scenario_name = st.sidebar.selectbox("Chọn kịch bản chính sách (2026-2030):", list(scenario_dict.keys()))
total_budget = st.sidebar.number_input("Tổng ngân sách (Tỷ VND):", min_value=50000, max_value=200000, value=100000, step=10000)

# Xử lý Logic Kịch bản
if scenario_dict[scenario_name] == "OPTIMIZE":
    with st.spinner('Đang chạy Bộ giải quy hoạch tuyến tính (CBC Solver)...'):
        ratios = run_m3_optimization(total_budget)
        st.sidebar.success("Đã tìm thấy phương án tối ưu toàn cục!")
else:
    ratios = scenario_dict[scenario_name]

# Chạy các hàm lõi
GDP_2030, budget_alloc = run_m1_forecasting(ratios, total_budget)
displaced_arr, upgrade_arr, netjob_arr = run_m4_labor(budget_alloc[2], budget_alloc[3])
total_netjob = np.sum(netjob_arr)

# --- KHU VỰC TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng quan", "💰 Phân bổ Ngân sách", "👥 Tác động Lao động", "⚖️ So sánh Kịch bản"])

with tab1:
    st.header(f"Chỉ số Vĩ mô 2030 - {scenario_name}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dự báo GDP 2030", f"{GDP_2030:,.0f} Tỷ", f"+{((GDP_2030/12847.6)-1)*100:.1f}% so vs 2025")
    col2.metric("Ngân sách AI & Nhân lực", f"{(budget_alloc[2]+budget_alloc[3]):,.0f} Tỷ", f"{ratios[2]+ratios[3]:.0%} tổng vốn")
    
    delta_color = "normal" if total_netjob >= 0 else "inverse"
    col3.metric("Việc làm ròng (NetJob)", f"{total_netjob:,.0f} Jobs", delta_color=delta_color)
    
    st.info("💡 **Góc nhìn chính sách:** GDP tăng trưởng dựa trên hàm Cobb-Douglas (M1), phụ thuộc chặt chẽ vào mức độ chuyển dịch từ vốn vật chất (K) sang công nghệ vô hình (AI/D).")

with tab2:
    st.header("Cơ cấu Phân bổ Nguồn lực")
    df_pie = pd.DataFrame({
        "Hạng mục": ["Vốn vật chất (K)", "Hạ tầng số (D)", "Trí tuệ Nhân tạo (AI)", "Nhân lực số (H)"],
        "Tỷ trọng": ratios,
        "Ngân sách": budget_alloc
    })
    
    col_pie, col_table = st.columns([3, 2])
    with col_pie:
        fig_pie = px.pie(df_pie, values='Tỷ trọng', names='Hạng mục', hole=0.4, title="Tỷ trọng Đầu tư")
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_table:
        st.dataframe(df_pie.style.format({'Tỷ trọng': '{:.1%}', 'Ngân sách': '{:,.0f}'}))

with tab3:
    st.header("Mô phỏng Dịch chuyển Việc làm (M4)")
    st.write("Sơ đồ luồng (Sankey) phân tích dòng chảy lao động bị tự động hóa so với năng lực nâng cấp kỹ năng.")
    
    # Chỉ lấy 3 ngành tiêu biểu để biểu đồ không bị rối
    target_indices = [1, 3, 4] # CN Chế biến, Bán lẻ, Tài chính
    source_labels = [sectors[i] for i in target_indices]
    target_labels = ["Bị đào thải (Displaced)", "Nâng cấp (Upgraded)", "Việc làm ròng mới"]
    
    # Lắp data Sankey
    sources = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    targets = [3, 4, 5, 3, 4, 5, 3, 4, 5]
    values = [
        displaced_arr[1], upgrade_arr[1], max(0, netjob_arr[1]),
        displaced_arr[3], upgrade_arr[3], max(0, netjob_arr[3]),
        displaced_arr[4], upgrade_arr[4], max(0, netjob_arr[4])
    ]
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=source_labels + target_labels),
        link=dict(source=sources, target=targets, value=values)
    )])
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    # Cảnh báo rủi ro (M5)
    st.subheader("Cảnh báo An sinh (M5)")
    for i in range(8):
        if displaced_arr[i] > (0.05 * L_jobs[i]):
            st.error(f"⚠️ Ngành **{sectors[i]}** có rủi ro mất việc làm vượt ngưỡng 5% tổng lao động ngành!")

with tab4:
    st.header("Bảng Tổng hợp So sánh")
    st.write("Chạy trước dữ liệu mô phỏng để ra quyết định dựa trên dữ liệu (Data-driven).")
    
    # Giả lập chạy cả 5 kịch bản để gom vào bảng
    compare_data = []
    for s_name, s_ratio in scenario_dict.items():
        r = run_m3_optimization(total_budget) if s_ratio == "OPTIMIZE" else s_ratio
        gdp, alloc = run_m1_forecasting(r, total_budget)
        disp, upg, nj = run_m4_labor(alloc[2], alloc[3])
        compare_data.append({
            "Kịch bản": s_name,
            "GDP 2030": round(gdp, 0),
            "% Vốn K": f"{r[0]:.0%}",
            "% Vốn AI": f"{r[2]:.0%}",
            "Việc làm ròng": round(np.sum(nj), 0)
        })
        
    df_compare = pd.DataFrame(compare_data)
    st.dataframe(df_compare, use_container_width=True)