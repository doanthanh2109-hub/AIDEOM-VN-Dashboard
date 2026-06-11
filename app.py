import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pulp

# ==========================================
# CẤU HÌNH TRANG & CSS TÙY CHỈNH
# ==========================================
st.set_page_config(page_title="VN AIDEOM-VN", layout="wide", page_icon="🇻🇳")

# Tùy chỉnh CSS để màu sắc metric giống hệt trong ảnh (Màu đỏ cho số liệu, xanh cho delta)
st.markdown("""
<style>
div[data-testid="metric-container"] > div > div > div {
    color: #ff4b4b; /* Màu đỏ cho giá trị chính */
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR NAVIGATION)
# ==========================================
menu_items = [
    "🏠 Trang chủ", 
    "📈 Bài 1 — Cobb-Douglas + AI", 
    "💰 Bài 2 — LP ngân sách số", 
    "📊 Bài 3 — Priority 10 ngành", 
    "🗺️ Bài 4 — LP ngành-vùng", 
    "🎯 Bài 5 — MIP 15 dự án", 
    "🏆 Bài 6 — TOPSIS 6 vùng", 
    "⚖️ Bài 7 — NSGA-II Pareto", 
    "⏳ Bài 8 — Động 2026-2035", 
    "👥 Bài 9 — Lao động & AI", 
    "🎲 Bài 10 — Stochastic SP", 
    "🤖 Bài 11 — Q-learning RL", 
    "🇻🇳 Bài 12 — AIDEOM tích hợp"
]

st.sidebar.markdown("### Danh mục")
choice = st.sidebar.radio("Điều hướng", menu_items, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**VN AIDEOM-VN**")
st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỷ nguyên AI")
st.sidebar.caption("📁 Dữ liệu: NSO, MoST, MIC, MPI, WB, GII")

# ==========================================
# TRANG 1: TRANG CHỦ (Tương ứng với hình ảnh)
# ==========================================
if choice == "🏠 Trang chủ":
    st.title("VN AIDEOM-VN")
    st.subheader("AI-Driven Decision Optimization Model for Vietnam")
    st.markdown("Web app giải 12 bài toán mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI — dữ liệu thực 2020-2025.")
    
    st.markdown("---")
    
    # Hiển thị 4 KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GDP 2025", "514,0 tỷ USD", "↑ +8,02%")
    col2.metric("Kinh tế số / GDP", "≈19,5%", "↑ +1,2 dpt")
    col3.metric("FDI giải ngân 2025", "27,6 tỷ USD", "↑ +8,9%")
    col4.metric("GDP/người 2025", "5.026 USD", "↑ +6,9%")
    
    st.markdown("---")
    st.markdown("### 📚 12 bài toán theo 4 cấp độ")
    
    # Expanders cho các cấp độ
    with st.expander("🟢 Cấp độ DỄ — Làm quen mô hình", expanded=True):
        st.markdown("""
        * **Bài 1:** Hàm sản xuất Cobb-Douglas mở rộng + AI — Growth accounting, dự báo GDP 2030 [cite: 13]
        * **Bài 2:** LP phân bổ ngân sách 4 hạng mục — scipy.optimize, shadow price [cite: 13]
        * **Bài 3:** Chỉ số ưu tiên 10 ngành — Min-max norm, weighted scoring, sensitivity [cite: 13]
        """)
        
    with st.expander("🟡 Cấp độ TRUNG BÌNH — Tối ưu cổ điển", expanded=True):
        st.markdown("""
        * **Bài 4:** LP phân bổ ngân sách số ngành-vùng — 24 biến, ràng buộc công bằng vùng [cite: 214]
        * **Bài 5:** MIP lựa chọn 15 dự án CĐS quốc gia — Knapsack + ràng buộc tiên quyết, PuLP/CBC [cite: 276]
        * **Bài 6:** TOPSIS xếp hạng 6 vùng — Trọng số chuyên gia + Entropy [cite: 338]
        """)
        
    with st.expander("🟠 Cấp độ KHÁ KHÓ — Đa mục tiêu & Động", expanded=False):
        st.markdown("""
        * **Bài 7:** Tối ưu đa mục tiêu Pareto với NSGA-II — pymoo, 4 mục tiêu xung đột
        * **Bài 8:** Tối ưu động phân bổ liên thời gian 2026-2035 — Quy hoạch phi tuyến (DP)
        * **Bài 9:** Tác động AI tới thị trường lao động — Mô phỏng NetJob, ngưỡng đào tạo lại
        """)
        
    with st.expander("🔴 Cấp độ KHÓ — Bất định & Tích hợp", expanded=False):
        st.markdown("""
        * **Bài 10:** Quy hoạch ngẫu nhiên hai giai đoạn (Stochastic SP) — Pyomo, GLPK
        * **Bài 11:** Học tăng cường (Q-learning) — Gymnasium, huấn luyện chính sách MDP
        * **Bài 12:** Đồ án tích hợp — Xây dựng nguyên mẫu AIDEOM-VN với Streamlit Dashboard
        """)
# ==========================================
# TRANG 2 -> 11: PLACEHOLDERS (Để trống cho tương lai)
# ==========================================
elif choice != "🇻🇳 Bài 12 — AIDEOM tích hợp":
    st.title(choice)
    st.info("Tính năng đang được phát triển. Bạn có thể chèn code của các bài tập tương ứng vào đây trong tương lai để tạo thành một Portfolio hoàn chỉnh.")

# ==========================================
# TRANG 12: DASHBOARD AIDEOM-VN (Code cũ)
# ==========================================
elif choice == "🇻🇳 Bài 12 — AIDEOM tích hợp":
    
    # --- DATA BÀI 1 & BÀI 9 ---
    alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
    K_2025, L_2025, D_2025, AI_2025, H_2025 = 25900, 53.4, 19.5, 80.1, 29.2
    A_bar = 0.98  
    
    sectors = ["Nông-Lâm", "CN chế tạo", "Xây dựng", "Bán lẻ", "Tài chính", "Logistics", "CNTT", "Giáo dục"]
    L_jobs = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15]) * 1_000_000
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0])
    c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    
    @st.cache_data
    def run_m3_optimization(total_budget=100000):
        beta_matrix = np.array([
            [1.15, 0.85, 0.55, 1.30], [0.95, 1.25, 1.40, 1.05],
            [1.05, 0.95, 0.85, 1.15], [1.20, 0.75, 0.45, 1.35],
            [0.90, 1.30, 1.55, 1.00], [1.10, 0.85, 0.65, 1.25]
        ])
        prob = pulp.LpProblem("S5_Optimization", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", ((r, j) for r in range(6) for j in range(4)), lowBound=0)
        prob += pulp.lpSum(beta_matrix[r, j] * x[r, j] for r in range(6) for j in range(4))
        prob += pulp.lpSum(x[r, j] for r in range(6) for j in range(4)) <= total_budget
        for r in range(6):
            prob += pulp.lpSum(x[r, j] for j in range(4)) >= total_budget * 0.05 
            prob += pulp.lpSum(x[r, j] for j in range(4)) <= total_budget * 0.30 
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        item_totals = [sum(x[r, j].varValue for r in range(6)) for j in range(4)]
        return [val / sum(item_totals) for val in item_totals] if sum(item_totals) > 0 else [0.25]*4

    def run_m1_forecasting(ratios, total_budget=100000):
        budget_alloc = [r * total_budget for r in ratios]
        K_2030 = K_2025 + budget_alloc[0] * 0.5 
        L_2030 = L_2025 * (1.01 ** 5)
        D_2030 = D_2025 + budget_alloc[1] * 0.001
        AI_2030 = AI_2025 + budget_alloc[2] * 0.005
        H_2030 = H_2025 + budget_alloc[3] * 0.002
        A_2030 = A_bar * (1.012 ** 5)
        GDP_2030 = A_2030 * (K_2030**alpha) * (L_2030**beta) * (D_2030**gamma) * (AI_2030**delta) * (H_2030**theta)
        return GDP_2030, budget_alloc

    def run_m4_labor(ai_budget, h_budget):
        x_AI = np.ones(8) * (ai_budget / 8)
        x_H = np.ones(8) * (h_budget / 8)
        return c1 * risk * x_AI, b1 * x_H, (a1 * x_AI) + (b1 * x_H) - (c1 * risk * x_AI)

    # --- UI BÀI 12 ---
    st.title("AIDEOM-VN: Hệ thống Hỗ trợ Ra quyết định")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Tùy chỉnh Kịch bản")
    scenario_dict = {
        "S1. Truyền thống": [0.70, 0.10, 0.10, 0.10],
        "S2. Số hóa nhanh": [0.25, 0.45, 0.15, 0.15],
        "S3. AI dẫn dắt": [0.20, 0.20, 0.45, 0.15],
        "S4. Bao trùm số": [0.30, 0.20, 0.10, 0.40],
        "S5. Tối ưu cân bằng": "OPTIMIZE"
    }

    scenario_name = st.sidebar.selectbox("Chọn kịch bản chính sách (2026-2030):", list(scenario_dict.keys()))
    total_budget = st.sidebar.number_input("Tổng ngân sách (Tỷ VND):", min_value=50000, value=100000, step=10000)

    if scenario_dict[scenario_name] == "OPTIMIZE":
        ratios = run_m3_optimization(total_budget)
    else:
        ratios = scenario_dict[scenario_name]

    GDP_2030, budget_alloc = run_m1_forecasting(ratios, total_budget)
    displaced_arr, upgrade_arr, netjob_arr = run_m4_labor(budget_alloc[2], budget_alloc[3])

    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan", "💰 Phân bổ Ngân sách", "👥 Tác động Lao động"])

    with tab1:
        st.header(f"Chỉ số Vĩ mô 2030 - {scenario_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Dự báo GDP 2030", f"{GDP_2030:,.0f} Tỷ")
        c2.metric("Ngân sách AI & Nhân lực", f"{(budget_alloc[2]+budget_alloc[3]):,.0f} Tỷ")
        c3.metric("Việc làm ròng (NetJob)", f"{np.sum(netjob_arr):,.0f} Jobs")

    with tab2:
        df_pie = pd.DataFrame({"Hạng mục": ["Vốn (K)", "Hạ tầng (D)", "AI", "Nhân lực (H)"], "Tỷ trọng": ratios})
        st.plotly_chart(px.pie(df_pie, values='Tỷ trọng', names='Hạng mục', hole=0.4), use_container_width=True)

    with tab3:
        sources = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        targets = [3, 4, 5, 3, 4, 5, 3, 4, 5]
        values = [displaced_arr[1], upgrade_arr[1], max(0, netjob_arr[1]),
                  displaced_arr[3], upgrade_arr[3], max(0, netjob_arr[3]),
                  displaced_arr[4], upgrade_arr[4], max(0, netjob_arr[4])]
        
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(label=[sectors[1], sectors[3], sectors[4], "Bị đào thải", "Nâng cấp", "Việc mới"]),
            link=dict(source=sources, target=targets, value=values)
        )])
        st.plotly_chart(fig_sankey, use_container_width=True)
