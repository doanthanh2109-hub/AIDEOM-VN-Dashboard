"""
VN AIDEOM-VN — Ứng dụng Streamlit tích hợp đầy đủ
Bao gồm: Trang chủ + Bài 1–11 + Bài 12 (Dashboard AIDEOM tích hợp)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# CẤU HÌNH TRANG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="VN AIDEOM-VN",
    layout="wide",
    page_icon="🇻🇳",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Metric value màu đỏ đặc trưng */
div[data-testid="metric-container"] > div > div > div {
    color: #e63946;
    font-weight: 700;
}
/* Sidebar nav gọn */
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
/* Tab header */
.stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
menu_items = [
    "🏠 Trang chủ",
    "📈 Bài 1 — Cobb-Douglas + AI",
    "💰 Bài 2 — LP ngân sách số",
    "📊 Bài 3 — Priority 10 ngành",
    "🗺️  Bài 4 — LP ngành-vùng",
    "🎯 Bài 5 — MIP lao động 2 GĐ",
    "🏆 Bài 6 — Đa mục tiêu vùng",
    "⚖️  Bài 7 — NSGA-II Pareto",
    "⏳ Bài 8 — Động 2026-2035",
    "👥 Bài 9 — Lao động & AI",
    "🎲 Bài 10 — Stochastic SP",
    "🤖 Bài 11 — Q-learning RL",
    "🇻🇳 Bài 12 — AIDEOM tích hợp",
]

st.sidebar.markdown("## 🇻🇳 VN AIDEOM-VN")
st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỷ nguyên AI")
st.sidebar.markdown("---")
choice = st.sidebar.radio("Điều hướng", menu_items, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("📁 Dữ liệu: GSO, MoST, MIC, MPI, WB, GII")


# ══════════════════════════════════════════════════════════════════
# TRANG CHỦ
# ══════════════════════════════════════════════════════════════════
if choice == "🏠 Trang chủ":
    st.title("VN AIDEOM-VN")
    st.subheader("AI-Driven Decision Optimization Model for Vietnam")
    st.markdown(
        "Web app giải **12 bài toán** mô hình ra quyết định phát triển kinh tế Việt Nam "
        "trong kỷ nguyên AI — dữ liệu thực 2020-2025."
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GDP 2025", "514,0 tỷ USD", "↑ +8,02%")
    col2.metric("Kinh tế số / GDP", "≈19,5%", "↑ +1,2 dpt")
    col3.metric("FDI giải ngân 2025", "27,6 tỷ USD", "↑ +8,9%")
    col4.metric("GDP/người 2025", "5.026 USD", "↑ +6,9%")

    st.markdown("---")
    st.markdown("### 📚 12 bài toán theo 4 cấp độ")

    with st.expander("🟢 Cấp độ DỄ — Làm quen mô hình", expanded=True):
        st.markdown("""
        * **Bài 1:** Hàm sản xuất Cobb-Douglas mở rộng + AI — Growth accounting, dự báo GDP 2030
        * **Bài 2:** LP phân bổ ngân sách 4 hạng mục — scipy.optimize, shadow price
        * **Bài 3:** Chỉ số ưu tiên 10 ngành — Min-max norm, weighted scoring, sensitivity
        """)
    with st.expander("🟡 Cấp độ TRUNG BÌNH — Tối ưu cổ điển", expanded=True):
        st.markdown("""
        * **Bài 4:** LP phân bổ ngân sách số ngành-vùng — 24 biến, ràng buộc công bằng vùng
        * **Bài 5:** Mô hình ngẫu nhiên 2 giai đoạn lao động — Knapsack + PuLP/CBC
        * **Bài 6:** Tối ưu đa mục tiêu 6 vùng — Goal Programming, chuẩn hóa Payoff
        """)
    with st.expander("🟠 Cấp độ KHÁ KHÓ — Đa mục tiêu & Động", expanded=False):
        st.markdown("""
        * **Bài 7:** Tối ưu đa mục tiêu Pareto với NSGA-II — pymoo, 4 mục tiêu xung đột
        * **Bài 8:** Tối ưu động phân bổ liên thời gian 2026-2035 — Quy hoạch phi tuyến (DP/CVXPY)
        * **Bài 9:** Tác động AI tới thị trường lao động — CVXPY, Sankey diagram
        """)
    with st.expander("🔴 Cấp độ KHÓ — Bất định & Tích hợp", expanded=False):
        st.markdown("""
        * **Bài 10:** Quy hoạch ngẫu nhiên hai giai đoạn (Stochastic SP) — Pyomo/HiGHS, VSS, EVPI
        * **Bài 11:** Học tăng cường (Q-learning) — Gymnasium, huấn luyện chính sách MDP
        * **Bài 12:** Đồ án tích hợp AIDEOM-VN — Dashboard Streamlit đa kịch bản
        """)


# ══════════════════════════════════════════════════════════════════
# BÀI 1 — COBB-DOUGLAS + AI
# ══════════════════════════════════════════════════════════════════
elif choice == "📈 Bài 1 — Cobb-Douglas + AI":
    st.header("Bài 1. Hàm sản xuất Cobb-Douglas mở rộng + AI")
    st.markdown("Phân tích đóng góp tăng trưởng (Growth Accounting) và dự báo GDP 2030.")

    @st.cache_data
    def calc_bai1():
        Year = np.array([2020, 2021, 2022, 2023, 2024, 2025])
        Y    = np.array([8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6])
        K    = np.array([16500, 17800, 19600, 21300, 23500, 25900])
        L    = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
        D    = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
        AI   = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
        H    = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])
        α, β, γ, δ, θ = 0.33, 0.42, 0.10, 0.08, 0.07

        data = pd.DataFrame({"Year": Year, "Y": Y, "K": K, "L": L, "D": D, "AI": AI, "H": H})
        data["A"] = Y / (K**α * L**β * D**γ * AI**δ * H**θ)
        A_bar = data["A"].mean()
        data["GDP_pred"] = A_bar * (K**α * L**β * D**γ * AI**δ * H**θ)
        data["APE"] = np.abs(Y - data["GDP_pred"]) / Y * 100
        MAPE = data["APE"].mean()

        growth = pd.DataFrame()
        growth["Period"]    = [f"{Year[i-1]}-{Year[i]}" for i in range(1, len(data))]
        growth["dlnY"]      = np.diff(np.log(Y))
        growth["Capital_K"] = α * np.diff(np.log(K))
        growth["Labor_L"]   = β * np.diff(np.log(L))
        growth["Digital_D"] = γ * np.diff(np.log(D))
        growth["AI"]        = δ * np.diff(np.log(AI))
        growth["Human_H"]   = θ * np.diff(np.log(H))
        growth["TFP"]       = np.diff(np.log(data["A"]))

        avg = {
            "Vốn (K)": growth["Capital_K"].mean(),
            "Lao động (L)": growth["Labor_L"].mean(),
            "Hạ tầng số (D)": growth["Digital_D"].mean(),
            "AI": growth["AI"].mean(),
            "Nhân lực số (H)": growth["Human_H"].mean(),
            "TFP": growth["TFP"].mean(),
        }
        total = sum(avg.values())
        contrib_df = pd.DataFrame({
            "Nhân tố": list(avg.keys()),
            "Đóng góp (%)": [v / total * 100 for v in avg.values()],
        })

        yrs = 5
        K2030  = K[-1] * (1.06 ** yrs)
        L2030  = L[-1] * (1.01 ** yrs)
        D2030, AI2030, H2030 = 30.0, 100.0, 35.0
        A2030  = data["A"].iloc[-1] * (1.012 ** yrs)
        GDP2030 = A2030 * K2030**α * L2030**β * D2030**γ * AI2030**δ * H2030**θ

        return data, MAPE, growth, contrib_df, GDP2030, K2030, L2030, D2030, AI2030, H2030, A2030

    data, MAPE, growth, contrib_df, GDP2030, K2030, L2030, D2030, AI2030, H2030, A2030 = calc_bai1()

    tab1, tab2, tab3 = st.tabs(["📊 TFP & Dự báo", "📈 Growth Accounting", "🔮 Kịch bản 2030"])

    with tab1:
        st.subheader("Năng suất nhân tố tổng hợp (TFP)")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(data[["Year", "A"]].style.format({"A": "{:.4f}"}), use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(data["Year"], data["A"], marker="o", linewidth=2, color="#1f77b4")
            ax.set_title("Total Factor Productivity (TFP) qua các năm")
            ax.set_xlabel("Năm"); ax.grid(True, alpha=0.4)
            st.pyplot(fig); plt.close(fig)

        st.subheader("GDP Thực tế vs Dự báo")
        st.metric("Sai số trung bình (MAPE)", f"{MAPE:.4f}%")
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        ax2.plot(data["Year"], data["Y"],        marker="o", label="Thực tế", color="#ff7f0e")
        ax2.plot(data["Year"], data["GDP_pred"], marker="s", label="Dự báo",  color="#2ca02c")
        ax2.set_ylabel("Nghìn tỷ VND"); ax2.legend(); ax2.grid(True, alpha=0.4)
        st.pyplot(fig2); plt.close(fig2)

    with tab2:
        st.subheader("Phân rã Tăng trưởng (Growth Accounting) 2020-2025")
        num_cols = [c for c in growth.columns if c != "Period"]
        st.dataframe(
            growth.style.format({c: "{:.4f}" for c in num_cols}),
            use_container_width=True,
        )
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        colors = ["#4c72b0","#dd8452","#55a868","#c44e52","#8172b2","#937860"]
        ax3.bar(contrib_df["Nhân tố"], contrib_df["Đóng góp (%)"], color=colors)
        ax3.set_title("Đóng góp vào tăng trưởng GDP 2020-2025 (%)")
        ax3.set_ylabel("%"); ax3.grid(axis="y", alpha=0.4)
        st.pyplot(fig3); plt.close(fig3)

    with tab3:
        st.subheader("Dự báo GDP năm 2030")
        ca, cb = st.columns(2)
        ca.metric("Vốn K_2030 (nghìn tỷ)", f"{K2030:,.0f}")
        ca.metric("Lao động L_2030 (triệu)", f"{L2030:.2f}")
        ca.metric("Hạ tầng số D_2030", f"{D2030}")
        cb.metric("Năng lực AI_2030", f"{AI2030}")
        cb.metric("Nhân lực số H_2030", f"{H2030}")
        cb.metric("TFP A_2030", f"{A2030:.4f}")
        st.success(f"🔥 **GDP dự báo 2030:** {GDP2030:,.2f} nghìn tỷ VND")


# ══════════════════════════════════════════════════════════════════
# BÀI 2 — LP NGÂN SÁCH SỐ
# ══════════════════════════════════════════════════════════════════
elif choice == "💰 Bài 2 — LP ngân sách số":
    st.header("Bài 2. Tối ưu phân bổ ngân sách chuyển đổi số (LP)")
    st.markdown("""
    **Bài toán:** Phân bổ 100 tỷ VND vào 4 hạng mục CĐS để tối đa hoá GDP gia tăng.

    | Biến | Hạng mục | Hệ số GDP | Sàn |
    |------|----------|-----------|-----|
    | x₁ | Hạ tầng số | 0,85 | ≥ 25 |
    | x₂ | AI & Dữ liệu | 1,20 | ≥ 15 |
    | x₃ | Nhân lực số | 0,95 | ≥ 20 |
    | x₄ | R&D công nghệ | 1,35 | ≥ 10 |
    """)

    @st.cache_data
    def calc_bai2():
        from scipy.optimize import linprog
        c_obj = [-0.85, -1.20, -0.95, -1.35]
        A_ub = [
            [1, 1, 1, 1],
            [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1],
            [0.35, -0.65, 0.35, -0.65],
        ]
        b_ub = [100, -25, -15, -20, -10, 0]
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*4, method="highs")
        x_opt = res.x
        Z_opt = -res.fun
        duals = res.ineqlin.marginals

        budgets = [100, 110, 120, 130, 140]
        z_sens = []
        for B in budgets:
            b_t = [B, -25, -15, -20, -10, 0]
            r2 = linprog(c_obj, A_ub=A_ub, b_ub=b_t, bounds=[(0, None)]*4, method="highs")
            z_sens.append(-r2.fun)

        # Kịch bản ưu tiên nhân lực (x3 >= 30)
        b_new = [100, -25, -15, -30, -10, 0]
        res_new = linprog(c_obj, A_ub=A_ub, b_ub=b_new, bounds=[(0, None)]*4, method="highs")
        return x_opt, Z_opt, duals, budgets, z_sens, res_new

    x_opt, Z_opt, duals, budgets, z_sens, res_new = calc_bai2()

    tab1, tab2, tab3 = st.tabs(["✅ Kết quả tối ưu", "🔍 Shadow Price", "📉 Phân tích độ nhạy"])

    with tab1:
        cols = st.columns(4)
        labels = ["x₁ Hạ tầng số", "x₂ AI & Dữ liệu", "x₃ Nhân lực số", "x₄ R&D"]
        for i, col in enumerate(cols):
            col.metric(labels[i], f"{x_opt[i]:.2f} tỷ")
        st.success(f"**Giá trị mục tiêu tối ưu Z* = {Z_opt:.2f} tỷ VND GDP gia tăng**")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(["Hạ tầng số", "AI & Dữ liệu", "Nhân lực số", "R&D"], x_opt,
               color=["#4c72b0","#dd8452","#55a868","#c44e52"])
        ax.set_ylabel("Ngân sách (tỷ VND)"); ax.set_title("Phương án phân bổ tối ưu")
        ax.grid(axis="y", alpha=0.4)
        st.pyplot(fig); plt.close(fig)

        if res_new.success:
            st.info(f"📌 **Kịch bản ưu tiên nhân lực (x₃ ≥ 30):** Z = {-res_new.fun:.2f} tỷ VND")
        else:
            st.warning("Kịch bản ưu tiên nhân lực (x₃ ≥ 30): Bài toán không khả thi.")

    with tab2:
        c_names = ["Tổng ngân sách", "x₁ ≥ 25", "x₂ ≥ 15", "x₃ ≥ 20", "x₄ ≥ 10", "Ràng buộc chiến lược"]
        df_dual = pd.DataFrame({"Ràng buộc": c_names, "Shadow Price": duals})
        st.dataframe(df_dual.style.format({"Shadow Price": "{:.6f}"}), use_container_width=True)
        st.markdown("""
        **Diễn giải Shadow Price:**
        - **Tổng ngân sách:** Tăng thêm 1 tỷ VND ngân sách → GDP gia tăng thêm đúng giá trị này.
        - **Ràng buộc sàn âm:** Giá trị âm cho thấy ràng buộc đang "kìm hãm" mục tiêu.
        """)

    with tab3:
        df_sens = pd.DataFrame({"Ngân sách (tỷ)": budgets, "Z tối ưu": z_sens})
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(df_sens.style.format({"Z tối ưu": "{:.2f}"}), use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(budgets, z_sens, marker="o", linewidth=2, color="#1f77b4")
            ax.set_xlabel("Ngân sách (tỷ VND)")
            ax.set_ylabel("GDP gia tăng tối ưu (tỷ)")
            ax.set_title("Sensitivity Analysis — Ngân sách vs GDP Gain")
            ax.grid(True, alpha=0.4)
            st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# BÀI 3 — PRIORITY 10 NGÀNH
# ══════════════════════════════════════════════════════════════════
elif choice == "📊 Bài 3 — Priority 10 ngành":
    st.header("Bài 3. Xếp hạng ngành ưu tiên chuyển đổi số")
    st.markdown("Chuẩn hóa min-max, tính điểm ưu tiên có trọng số và phân tích độ nhạy AI readiness.")

    @st.cache_data
    def calc_bai3():
        np.random.seed(42)
        sectors_vi = [
            "Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng",
            "Bán buôn bán lẻ", "Tài chính-Ngân hàng", "Logistics-Vận tải",
            "CNTT-Truyền thông", "Giáo dục-Đào tạo", "Y tế", "Du lịch"
        ]
        df = pd.DataFrame({
            "sector_name_vi": sectors_vi,
            "growth_rate_2024_pct": [3.2, 8.5, 7.1, 9.2, 12.4, 6.8, 18.5, 5.2, 7.8, 10.1],
            "gdp_share_2024_pct":   [11.5, 24.7, 5.8, 14.2, 8.5, 4.2, 3.8, 3.5, 2.8, 3.2],
            "spillover_coef_0_1":   [0.35, 0.72, 0.45, 0.68, 0.82, 0.58, 0.95, 0.65, 0.72, 0.55],
            "export_billion_USD":   [3.2, 25.8, 0.8, 4.5, 0.5, 2.8, 8.5, 0.3, 0.2, 6.8],
            "labor_million":        [13.2, 11.5, 4.8, 7.8, 0.55, 1.95, 0.62, 2.15, 0.85, 1.2],
            "ai_readiness_0_100":   [35, 62, 48, 58, 82, 55, 92, 68, 72, 61],
            "automation_risk_pct":  [18, 42, 25, 38, 52, 35, 28, 22, 30, 40],
        })

        cols_good = ["growth_rate_2024_pct","gdp_share_2024_pct","spillover_coef_0_1",
                     "export_billion_USD","labor_million","ai_readiness_0_100"]
        Xg = df[cols_good].apply(lambda x: (x - x.min())/(x.max() - x.min()))
        Xb = (df["automation_risk_pct"].max() - df["automation_risk_pct"]) / \
             (df["automation_risk_pct"].max() - df["automation_risk_pct"].min())

        w_good = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20])
        w_risk = 0.15
        priority = Xg.values @ w_good + w_risk * Xb.values
        df["Priority"] = priority
        ranking = df[["sector_name_vi","Priority"]].sort_values("Priority", ascending=False).reset_index(drop=True)
        ranking.index += 1

        ai_weights = np.arange(0.05, 0.45, 0.05)
        heat_rows = {}
        for ai_w in ai_weights:
            base = np.array([0.15, 0.15, 0.20, 0.15, 0.10, ai_w, 0.15])
            base /= base.sum()
            score = Xg.values @ base[:6] + base[6] * Xb.values
            top3 = pd.Series(score, index=df["sector_name_vi"]).nlargest(3).index.tolist()
            heat_rows[f"{ai_w:.2f}"] = {s: (1 if s in top3 else 0) for s in df["sector_name_vi"]}
        heat_df = pd.DataFrame(heat_rows).T  # ai_weight × sector

        return df, ranking, Xg, Xb, heat_df

    df3, ranking3, Xg3, Xb3, heat_df3 = calc_bai3()

    tab1, tab2, tab3 = st.tabs(["🏅 Xếp hạng mặc định", "🌡️ Heatmap độ nhạy AI", "⚖️ So sánh 2 kịch bản"])

    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(ranking3.style.format({"Priority": "{:.4f}"}), use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(ranking3["sector_name_vi"][::-1], ranking3["Priority"][::-1], color="#4c72b0")
            ax.set_xlabel("Điểm ưu tiên"); ax.set_title("Xếp hạng ngành ưu tiên CĐS")
            ax.grid(axis="x", alpha=0.4)
            st.pyplot(fig); plt.close(fig)

    with tab2:
        st.markdown("**Heatmap:** Ô màu sáng = ngành thuộc Top-3 tại trọng số AI đó")
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(heat_df3.values.T, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(heat_df3.index)));   ax.set_xticklabels(heat_df3.index, fontsize=8)
        ax.set_yticks(range(len(heat_df3.columns))); ax.set_yticklabels(heat_df3.columns, fontsize=8)
        ax.set_xlabel("Trọng số AI Readiness"); ax.set_title("Độ nhạy Top-3 theo trọng số AI")
        plt.colorbar(im, ax=ax, fraction=0.02)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with tab3:
        w_growth   = np.array([0.25, 0.20, 0.10, 0.20, 0.05, 0.10]); wr_growth = 0.10
        w_incl     = np.array([0.10, 0.10, 0.25, 0.05, 0.20, 0.10]); wr_incl   = 0.20
        s_growth   = Xg3.values @ w_growth + wr_growth * Xb3.values
        s_incl     = Xg3.values @ w_incl   + wr_incl   * Xb3.values

        rank_g = pd.DataFrame({"Ngành": df3["sector_name_vi"], "Score Tăng trưởng": s_growth}) \
                   .sort_values("Score Tăng trưởng", ascending=False).reset_index(drop=True)
        rank_i = pd.DataFrame({"Ngành": df3["sector_name_vi"], "Score Bao trùm": s_incl}) \
                   .sort_values("Score Bao trùm", ascending=False).reset_index(drop=True)
        rank_g.index += 1; rank_i.index += 1

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📈 Định hướng Tăng trưởng**")
            st.dataframe(rank_g.head(5).style.format({"Score Tăng trưởng": "{:.4f}"}), use_container_width=True)
        with c2:
            st.markdown("**🤝 Định hướng Bao trùm**")
            st.dataframe(rank_i.head(5).style.format({"Score Bao trùm": "{:.4f}"}), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# BÀI 4 — LP NGÀNH-VÙNG
# ══════════════════════════════════════════════════════════════════
elif choice == "🗺️  Bài 4 — LP ngành-vùng":
    st.header("Bài 4. LP phân bổ ngân sách số ngành-vùng (có ràng buộc công bằng)")
    st.markdown("""
    Tối đa hoá **Tổng GDP Gain** khi phân bổ 50.000 tỷ VND vào **6 vùng × 4 hạng mục** (24 biến),
    có ràng buộc công bằng vùng miền (C5).
    """)

    try:
        import pulp

        @st.cache_data
        def calc_bai4():
            regions = ['Trung du MN Bắc', 'ĐB sông Hồng', 'BTBộ & DHMT', 'Tây Nguyên', 'Đông Nam Bộ', 'ĐB SCL']
            items   = ['I (Hạ tầng)', 'D (CĐS DN)', 'AI', 'H (Nhân lực)']
            beta    = np.array([
                [1.15, 0.85, 0.55, 1.30],
                [0.95, 1.25, 1.40, 1.05],
                [1.05, 0.95, 0.85, 1.15],
                [1.20, 0.75, 0.45, 1.35],
                [0.90, 1.30, 1.55, 1.00],
                [1.10, 0.85, 0.65, 1.25],
            ])
            D_init = np.array([38.0, 78.0, 55.0, 32.0, 82.0, 48.0])
            TOTAL, MIN_R, MAX_R, MIN_H, GAMMA, LAMBDA = 50000, 5000, 12000, 12000, 0.002, 0.7

            # ── Có công bằng C5 ──
            prob = pulp.LpProblem("Fair", pulp.LpMaximize)
            x = pulp.LpVariable.dicts("x", ((r,j) for r in range(6) for j in range(4)), lowBound=0)
            M = pulp.LpVariable("M")
            prob += pulp.lpSum(beta[r,j]*x[r,j] for r in range(6) for j in range(4))
            prob += pulp.lpSum(x[r,j] for r in range(6) for j in range(4)) <= TOTAL
            for r in range(6):
                prob += pulp.lpSum(x[r,j] for j in range(4)) >= MIN_R
                prob += pulp.lpSum(x[r,j] for j in range(4)) <= MAX_R
                prob += M >= D_init[r] + GAMMA * x[r,1]
                prob += D_init[r] + GAMMA * x[r,1] >= LAMBDA * M
            prob += pulp.lpSum(x[r,3] for r in range(6)) >= MIN_H
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            Z_fair = pulp.value(prob.objective)
            fair = np.array([[x[r,j].varValue for j in range(4)] for r in range(6)])

            # ── Không công bằng ──
            prob2 = pulp.LpProblem("Unfair", pulp.LpMaximize)
            xu = pulp.LpVariable.dicts("xu", ((r,j) for r in range(6) for j in range(4)), lowBound=0)
            prob2 += pulp.lpSum(beta[r,j]*xu[r,j] for r in range(6) for j in range(4))
            prob2 += pulp.lpSum(xu[r,j] for r in range(6) for j in range(4)) <= TOTAL
            for r in range(6):
                prob2 += pulp.lpSum(xu[r,j] for j in range(4)) >= MIN_R
                prob2 += pulp.lpSum(xu[r,j] for j in range(4)) <= MAX_R
            prob2 += pulp.lpSum(xu[r,3] for r in range(6)) >= MIN_H
            prob2.solve(pulp.PULP_CBC_CMD(msg=False))
            Z_unfair = pulp.value(prob2.objective)
            unfair = np.array([[xu[r,j].varValue for j in range(4)] for r in range(6)])

            return regions, items, fair, unfair, Z_fair, Z_unfair

        regions4, items4, fair4, unfair4, Z_fair4, Z_unfair4 = calc_bai4()

        tab1, tab2, tab3 = st.tabs(["📊 Kết quả phân bổ", "🌡️ Heatmap so sánh", "📐 Chi phí công bằng"])

        with tab1:
            c1, c2 = st.columns(2)
            c1.metric("Z* (Có ràng buộc C5)", f"{Z_fair4:,.2f} tỷ")
            c2.metric("Z* (Không có C5)",      f"{Z_unfair4:,.2f} tỷ")
            df_f = pd.DataFrame(fair4, index=regions4, columns=items4)
            st.dataframe(df_f.style.format("{:.1f}"), use_container_width=True)

        with tab2:
            try:
                import seaborn as sns
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                sns.heatmap(pd.DataFrame(fair4,   index=regions4, columns=items4),
                            annot=True, fmt=".0f", cmap="YlGnBu", ax=axes[0])
                axes[0].set_title("CÓ ràng buộc công bằng (C5)")
                sns.heatmap(pd.DataFrame(unfair4, index=regions4, columns=items4),
                            annot=True, fmt=".0f", cmap="OrRd", ax=axes[1])
                axes[1].set_title("KHÔNG có ràng buộc công bằng")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            except ImportError:
                st.info("Cài seaborn để xem heatmap: pip install seaborn")

        with tab3:
            cost = Z_unfair4 - Z_fair4
            st.metric("Chi phí kinh tế đánh đổi cho công bằng", f"{cost:,.2f} tỷ VND GDP Gain")
            st.markdown(f"""
            **Diễn giải:** Khi áp dụng ràng buộc công bằng vùng miền (C5), tổng GDP Gain giảm đi
            **{cost:,.2f} tỷ VND** so với kịch bản tập trung hoàn toàn vào hiệu quả.
            Đây là chi phí xã hội để đảm bảo không vùng nào bị bỏ lại quá xa.
            """)

    except ImportError:
        st.error("Cần cài đặt thư viện: pip install pulp")


# ══════════════════════════════════════════════════════════════════
# BÀI 5 — MIP LAO ĐỘNG 2 GIAI ĐOẠN
# ══════════════════════════════════════════════════════════════════
elif choice == "🎯 Bài 5 — MIP lao động 2 GĐ":
    st.header("Bài 5. Mô hình ngẫu nhiên 2 giai đoạn — An sinh lao động trước AI")
    st.markdown("""
    Tối thiểu hoá **Tổng chi phí kỳ vọng** = Chi phí đầu tư phòng ngừa GĐ1
    \+ Kỳ vọng(Chi phí đào tạo lại + Phạt thất nghiệp) GĐ2, với 2 kịch bản AI.
    """)

    try:
        import pulp as pl

        @st.cache_data
        def calc_bai5():
            sectors = [
                "Nông-Lâm-Thủy sản","CN chế biến chế tạo","Xây dựng","Bán buôn bán lẻ",
                "Tài chính-Ngân hàng","Logistics-Vận tải","CNTT-Truyền thông","Giáo dục-Đào tạo",
                "Y tế","Du lịch",
            ]
            labor_m  = np.array([13.2, 11.5, 4.8, 7.8, 0.55, 1.95, 0.62, 2.15, 0.85, 1.2])
            auto_r   = np.array([18, 42, 25, 38, 52, 35, 28, 22, 30, 40]) / 100
            ai_ready = np.array([35, 62, 48, 58, 82, 55, 92, 68, 72, 61])
            B = 10000; x_lo, x_hi = 500, 3000
            k = {s: 0.0005 * (ai_ready[i]/50.0) for i, s in enumerate(sectors)}
            scenarios = ["AI_Baseline","AI_Surge"]
            probs  = {"AI_Baseline": 0.70, "AI_Surge": 0.30}
            shocks = {"AI_Baseline": 1.0,  "AI_Surge": 1.8}
            q = {"AI_Baseline": 15, "AI_Surge": 25}
            d = {"AI_Baseline": 35, "AI_Surge": 65}

            risk_labor = {}
            for s in scenarios:
                risk_labor[s] = {sectors[i]: labor_m[i] * auto_r[i] * shocks[s] for i in range(len(sectors))}

            model = pl.LpProblem("2Stage_Social", pl.LpMinimize)
            x = pl.LpVariable.dicts("X", sectors, lowBound=x_lo, upBound=x_hi)
            y = pl.LpVariable.dicts("Y", (sectors, scenarios), lowBound=0)
            u = pl.LpVariable.dicts("U", (sectors, scenarios), lowBound=0)

            model += pl.lpSum(x[i] for i in sectors) + \
                     pl.lpSum(probs[s]*(pl.lpSum(q[s]*y[i][s]+d[s]*u[i][s] for i in sectors)) for s in scenarios)
            model += pl.lpSum(x[i] for i in sectors) <= B
            for s in scenarios:
                for i in sectors:
                    model += u[i][s] >= risk_labor[s][i] - k[i]*x[i] - y[i][s]
            model.solve(pl.PULP_CBC_CMD(msg=False))

            gd1 = pd.DataFrame([{"Ngành": i, "Đầu tư GĐ1 (tỷ)": round(x[i].varValue,2)} for i in sectors])
            rows = []
            for s in scenarios:
                for i in sectors:
                    rows.append({
                        "Kịch bản": s, "Ngành": i,
                        "Nguy cơ (triệu người)": round(risk_labor[s][i], 3),
                        "Đào tạo lại (triệu)": round(y[i][s].varValue, 3),
                        "Thất nghiệp tồn đọng (triệu)": round(u[i][s].varValue, 3),
                    })
            gd2 = pd.DataFrame(rows)
            total_cost = pl.value(model.objective)
            return gd1, gd2, total_cost, sectors, scenarios

        gd1, gd2, total_cost, sectors5, scenarios5 = calc_bai5()

        st.metric("Tổng chi phí kỳ vọng tối thiểu", f"{total_cost:,.2f} tỷ VND")
        tab1, tab2 = st.tabs(["💼 Phân bổ GĐ1", "📊 Kết quả GĐ2"])

        with tab1:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(gd1.style.format({"Đầu tư GĐ1 (tỷ)": "{:.2f}"}), use_container_width=True)
            with c2:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.barh(gd1["Ngành"], gd1["Đầu tư GĐ1 (tỷ)"], color="#4c72b0")
                ax.set_xlabel("Tỷ VND"); ax.set_title("Đầu tư phòng ngừa tối ưu (Giai đoạn 1)")
                ax.grid(axis="x", alpha=0.4); plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

        with tab2:
            for s in scenarios5:
                st.markdown(f"**Kịch bản: {s}**")
                df_s = gd2[gd2["Kịch bản"] == s].drop(columns="Kịch bản")
                st.dataframe(df_s.style.format({
                    "Nguy cơ (triệu người)": "{:.3f}",
                    "Đào tạo lại (triệu)": "{:.3f}",
                    "Thất nghiệp tồn đọng (triệu)": "{:.3f}",
                }), use_container_width=True)

    except ImportError:
        st.error("Cần cài đặt thư viện: pip install pulp")


# ══════════════════════════════════════════════════════════════════
# BÀI 6 — ĐA MỤC TIÊU 6 VÙNG
# ══════════════════════════════════════════════════════════════════
elif choice == "🏆 Bài 6 — Đa mục tiêu vùng":
    st.header("Bài 6. Tối ưu đa mục tiêu phân bổ ngân sách số 6 vùng")
    st.markdown("""
    **3 mục tiêu:** (1) Tối đa hoá GRDP tăng thêm · (2) Giảm thiểu bất bình đẳng (Gini)
    · (3) Tối đa hoá lan toả AI. Phương pháp **Weighted Sum** với chuẩn hóa Payoff Matrix.
    """)

    try:
        import pulp as pl

        @st.cache_data
        def calc_bai6():
            regions = ['Trung du MN Bắc','ĐB sông Hồng','BTBộ & DHMT','Tây Nguyên','Đông Nam Bộ','ĐB SCL']
            grdp   = np.array([185.0, 1850.0, 620.0, 145.0, 2850.0, 480.0])
            growth = np.array([7.2, 8.5, 6.8, 7.5, 8.2, 7.0])
            gini   = np.array([0.42, 0.38, 0.40, 0.45, 0.36, 0.41])
            ai_r   = np.array([35.0, 78.0, 52.0, 30.0, 85.0, 48.0])

            alpha = {regions[i]: grdp[i]*(growth[i]/100)/100 for i in range(6)}
            beta  = {regions[i]: gini[i] for i in range(6)}
            gamma = {regions[i]: ai_r[i]/100 for i in range(6)}

            B = 15000; x_min = 1000

            def solve_single(target, mode):
                p = pl.LpProblem("S", pl.LpMaximize if mode=="max" else pl.LpMinimize)
                xv = pl.LpVariable.dicts("x", regions, lowBound=x_min)
                p += pl.lpSum(xv[r] for r in regions) <= B
                if target=="growth": p += pl.lpSum(alpha[r]*xv[r] for r in regions)
                elif target=="gini": p += pl.lpSum(beta[r]*xv[r]  for r in regions)
                else:                p += pl.lpSum(gamma[r]*xv[r] for r in regions)
                p.solve(pl.PULP_CBC_CMD(msg=False))
                return pl.value(p.objective)

            f1_max,f1_min = solve_single("growth","max"), solve_single("growth","min")
            f2_max,f2_min = solve_single("gini","max"),   solve_single("gini","min")
            f3_max,f3_min = solve_single("ai","max"),     solve_single("ai","min")

            mo = pl.LpProblem("MO", pl.LpMaximize)
            xm = pl.LpVariable.dicts("xm", regions, lowBound=x_min)
            mo += pl.lpSum(xm[r] for r in regions) <= B
            f1e = pl.lpSum(alpha[r]*xm[r] for r in regions)
            f2e = pl.lpSum(beta[r]*xm[r]  for r in regions)
            f3e = pl.lpSum(gamma[r]*xm[r] for r in regions)
            w1,w2,w3 = 0.4, 0.3, 0.3
            denom1 = max(f1_max-f1_min,1); denom2 = max(f2_max-f2_min,1); denom3 = max(f3_max-f3_min,1)
            mo += w1*((f1e-f1_min)/denom1) - w2*((f2e-f2_min)/denom2) + w3*((f3e-f3_min)/denom3)
            mo.solve(pl.PULP_CBC_CMD(msg=False))

            alloc = {r: xm[r].varValue for r in regions}
            ff1 = sum(alpha[r]*alloc[r] for r in regions)
            ff2 = sum(beta[r]*alloc[r]  for r in regions)
            ff3 = sum(gamma[r]*alloc[r] for r in regions)
            return regions, alloc, ff1, ff2, ff3, f1_min,f1_max, f2_min,f2_max, f3_min,f3_max

        regions6, alloc6, ff1,ff2,ff3, f1lo,f1hi,f2lo,f2hi,f3lo,f3hi = calc_bai6()

        tab1, tab2 = st.tabs(["📊 Kết quả phân bổ", "📐 Đánh giá mục tiêu"])

        with tab1:
            df_a = pd.DataFrame({"Vùng": regions6, "Ngân sách (tỷ)": [alloc6[r] for r in regions6]})
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(df_a.style.format({"Ngân sách (tỷ)": "{:,.0f}"}), use_container_width=True)
            with c2:
                fig, ax = plt.subplots(figsize=(7, 4))
                bars = ax.barh(df_a["Vùng"], df_a["Ngân sách (tỷ)"], color="#55a868")
                for b in bars:
                    ax.text(b.get_width()+100, b.get_y()+b.get_height()/2,
                            f"{int(b.get_width()):,}", va="center", fontsize=9)
                ax.set_xlabel("Tỷ VND"); ax.set_title("Phân bổ ngân sách số tối ưu đa mục tiêu")
                ax.grid(axis="x", alpha=0.4); plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

        with tab2:
            data_muc = {
                "Mục tiêu": ["GRDP tăng thêm (f1)","Rủi ro Gini (f2 – minimize)","Lan toả AI (f3)"],
                "Đạt được": [ff1, ff2, ff3],
                "Lý tưởng min": [f1lo, f2lo, f3lo],
                "Lý tưởng max": [f1hi, f2hi, f3hi],
            }
            st.dataframe(pd.DataFrame(data_muc).style.format({
                "Đạt được":"{:.2f}","Lý tưởng min":"{:.2f}","Lý tưởng max":"{:.2f}"}),
                use_container_width=True)

    except ImportError:
        st.error("Cần cài đặt: pip install pulp")


# ══════════════════════════════════════════════════════════════════
# BÀI 7 — NSGA-II PARETO
# ══════════════════════════════════════════════════════════════════
elif choice == "⚖️  Bài 7 — NSGA-II Pareto":
    st.header("Bài 7. Tối ưu đa mục tiêu Pareto với NSGA-II")
    st.markdown("""
    **4 mục tiêu xung đột:** f1 Max GDP Gain · f2 Min Bất bình đẳng · f3 Min Phát thải · f4 Min Rủi ro ròng.
    Thuật toán **NSGA-II** (pymoo) tìm tập Pareto, sau đó **TOPSIS** chọn nghiệm thỏa hiệp.
    """)

    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize as pymoo_minimize

        @st.cache_data
        def calc_bai7():
            class Prob(ElementwiseProblem):
                def __init__(self):
                    super().__init__(n_var=24, n_obj=4, n_ieq_constr=12,
                                     xl=np.zeros(24), xu=np.ones(24)*12000)
                    self.beta = np.array([
                        [0.25,0.30,0.35,0.20],[0.22,0.28,0.33,0.25],
                        [0.20,0.25,0.30,0.22],[0.18,0.22,0.28,0.20],
                        [0.26,0.32,0.38,0.24],[0.15,0.20,0.25,0.18],
                    ])
                    self.e   = np.array([0.42,0.55,0.48,0.32,0.62,0.38])
                    self.rho = np.array([0.18,0.45,0.28,0.12,0.52,0.22])
                    self.sig = np.array([0.32,0.28,0.30,0.35,0.25,0.30])

                def _evaluate(self, x, out, *args, **kwargs):
                    X = x.reshape(6,4)
                    f1 = -(self.beta*X).sum()
                    sv = X.sum(axis=1)
                    f2 = np.abs(sv - sv.mean()).mean()
                    f3 = (self.e*(X[:,0]+X[:,2])).sum()
                    f4 = (self.rho*X[:,2]).sum()-(self.sig*X[:,3]).sum()
                    g  = np.zeros(12)
                    g[0] = X.sum()-35000
                    for i in range(6): g[1+i] = X[i,:].sum()-12000
                    sm = X.sum(axis=0)
                    for j in range(4): g[7+j] = 4000-sm[j]
                    g[11] = f3-15000
                    out['F'] = [f1,f2,f3,f4]; out['G'] = g

            prob = Prob()
            algo = NSGA2(pop_size=80)
            res  = pymoo_minimize(prob, algo, ('n_gen',150), seed=42, verbose=False)
            Fp = res.F.copy()
            Fp[:,0] = -Fp[:,0]   # f1 → positivo

            w = np.array([0.40,0.25,0.20,0.15])
            norm = res.F / np.sqrt((res.F**2).sum(axis=0))
            wm   = norm * w
            ib   = wm.min(axis=0); iw = wm.max(axis=0)
            db   = np.sqrt(((wm-ib)**2).sum(axis=1))
            dw   = np.sqrt(((wm-iw)**2).sum(axis=1))
            scores = dw/(db+dw)
            best   = np.argmax(scores)
            return Fp, scores, best

        with st.spinner("Đang chạy NSGA-II (≈ 30 giây)…"):
            Fp7, scores7, best7 = calc_bai7()

        tab1, tab2, tab3 = st.tabs(["🌌 Scatter 3D", "📊 Parallel Coords", "🎯 TOPSIS & Chi phí"])

        with tab1:
            fig = go.Figure(data=[go.Scatter3d(
                x=Fp7[:,0], y=Fp7[:,1], z=Fp7[:,2],
                mode="markers",
                marker=dict(size=4, color=Fp7[:,3], colorscale="Viridis", showscale=True,
                            colorbar=dict(title="f4 Rủi ro")),
            )])
            fig.update_layout(
                scene=dict(xaxis_title="f1 GDP Gain (Max)",
                           yaxis_title="f2 Bất bình đẳng (Min)",
                           zaxis_title="f3 Phát thải (Min)"),
                title=f"Tập Pareto NSGA-II — {len(Fp7)} nghiệm",
                height=550,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            Fn = (Fp7 - Fp7.min(axis=0)) / (Fp7.max(axis=0) - Fp7.min(axis=0) + 1e-9)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            labels7 = ["f1 GDP Gain","f2 Bất bình đẳng","f3 Phát thải","f4 Rủi ro"]
            for row in Fn[::max(1,len(Fn)//60)]:
                ax2.plot(labels7, row, color="teal", alpha=0.25, linewidth=0.8)
            ax2.plot(labels7, Fn[best7], color="red", linewidth=2.5, label="Nghiệm TOPSIS")
            ax2.legend(); ax2.grid(True, alpha=0.4)
            ax2.set_title("Parallel Coordinates — Đã chuẩn hóa (nghiệm đỏ = TOPSIS)")
            st.pyplot(fig2); plt.close(fig2)

        with tab3:
            sol = Fp7[best7]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("f1 GDP Gain (Max)",    f"{sol[0]:,.1f}")
            c2.metric("f2 Bất bình đẳng (Min)", f"{sol[1]:,.1f}")
            c3.metric("f3 Phát thải (Min)",     f"{sol[2]:,.1f}")
            c4.metric("f4 Rủi ro ròng (Min)",   f"{sol[3]:,.1f}")

            max_g = Fp7[np.argmax(Fp7[:,0])]
            st.markdown("**Phân tích chi phí cơ hội** (Max GDP vs Thỏa hiệp TOPSIS):")
            st.markdown(f"""
            | Chỉ số | Nghiệm Max GDP | Nghiệm Thỏa hiệp |
            |--------|---------------|------------------|
            | f1 GDP Gain | {max_g[0]:,.1f} | {sol[0]:,.1f} |
            | f2 Bất bình đẳng | {max_g[1]:,.1f} | {sol[1]:,.1f} |
            | f3 Phát thải | {max_g[2]:,.1f} | {sol[2]:,.1f} |
            """)

    except ImportError:
        st.warning("Cần cài đặt pymoo: `pip install pymoo`")
        st.markdown("Khi chạy đầy đủ, trang này hiển thị tập Pareto 3D và biểu đồ Parallel Coordinates.")


# ══════════════════════════════════════════════════════════════════
# BÀI 8 — ĐỘNG 2026-2035
# ══════════════════════════════════════════════════════════════════
elif choice == "⏳ Bài 8 — Động 2026-2035":
    st.header("Bài 8. Tối ưu động phân bổ liên thời gian 2026-2035")
    st.markdown("""
    Mô hình **Ramsey động** với CVXPY/SCS: tối đa hoá tổng hữu dụng phúc lợi chiết khấu
    $\\sum_{t=0}^{9} \\rho^t \\ln(C_t)$ với Cobb-Douglas 5 nhân tố.
    """)

    try:
        import cvxpy as cp

        @st.cache_data
        def calc_bai8():
            T = 10; rho = 0.97
            aK,aD,aAI,aH = 0.33,0.10,0.08,0.07
            aL = 1-(aK+aD+aAI+aH)
            dK,dD,dAI,dH = 0.05,0.12,0.15,0.02
            L_v,A_v = 54.0, 1.0
            K0,D0,AI0,H0 = 27500.0,20.3,86.0,30.0

            K  = cp.Variable(T+1,nonneg=True); D  = cp.Variable(T+1,nonneg=True)
            AI = cp.Variable(T+1,nonneg=True); H  = cp.Variable(T+1,nonneg=True)
            IK = cp.Variable(T,nonneg=True);   ID = cp.Variable(T,nonneg=True)
            IAI= cp.Variable(T,nonneg=True);   IH = cp.Variable(T,nonneg=True)
            lK = cp.Variable(T); lD = cp.Variable(T); lAI= cp.Variable(T); lH = cp.Variable(T)
            lY = cp.Variable(T); C  = cp.Variable(T,nonneg=True)

            cons = [K[0]==K0,D[0]==D0,AI[0]==AI0,H[0]==H0]
            for t in range(T):
                cons += [lK[t]<=cp.log(K[t]),lD[t]<=cp.log(D[t]),
                         lAI[t]<=cp.log(AI[t]),lH[t]<=cp.log(H[t])]
                cons += [lY[t] == np.log(A_v)+aK*lK[t]+aD*lD[t]+aAI*lAI[t]+aH*lH[t]+aL*np.log(L_v)]
                cons += [C[t]+IK[t]+ID[t]+IAI[t]+IH[t] <= cp.exp(lY[t])]
                cons += [K[t+1]==(1-dK)*K[t]+IK[t], D[t+1]==(1-dD)*D[t]+ID[t],
                         AI[t+1]==(1-dAI)*AI[t]+IAI[t], H[t+1]==(1-dH)*H[t]+0.8*IH[t]]

            prob = cp.Problem(cp.Maximize(sum(rho**t*cp.log(C[t]) for t in range(T))), cons)
            prob.solve(solver=cp.SCS, verbose=False)
            years = np.arange(2026, 2036)
            Y_v = np.exp(lY.value)
            return years, Y_v, C.value, K.value[:-1], D.value[:-1], AI.value[:-1], H.value[:-1], prob.value

        with st.spinner("Đang giải mô hình động CVXPY (≈ 20 giây)…"):
            yrs8, Y8, C8, K8, D8, AI8, H8, welfare8 = calc_bai8()

        st.metric("Tổng phúc lợi xã hội tích lũy (chiết khấu)", f"{welfare8:.4f}")

        df8 = pd.DataFrame({
            "Năm": yrs8, "GDP (Y)": np.round(Y8,2), "Tiêu dùng (C)": np.round(C8,2),
            "Vốn (K)": np.round(K8,2), "Hạ tầng số (D)": np.round(D8,3),
            "AI Score": np.round(AI8,2), "Nhân lực H": np.round(H8,3),
        })
        st.dataframe(df8.style.format({c:"{:.2f}" for c in df8.columns if c!="Năm"}), use_container_width=True)

        tab1, tab2 = st.tabs(["📈 Sản lượng & Tiêu dùng", "🔬 Vốn & AI & Nhân lực"])

        with tab1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(yrs8, Y8, marker="o", color="navy",       label="Sản lượng GDP (Y)")
            ax.plot(yrs8, C8, marker="s", color="darkorange", linestyle="--", label="Tiêu dùng (C)")
            ax.set_ylabel("Nghìn tỷ VND"); ax.legend(); ax.grid(True, alpha=0.4)
            ax.set_title("Quỹ đạo Sản lượng vs Tiêu dùng 2026-2035")
            st.pyplot(fig); plt.close(fig)

        with tab2:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(yrs8, K8, marker="^", color="green")
            axes[0].set_title("Vốn vật chất (K)"); axes[0].grid(True, alpha=0.4)
            ax2t = axes[1].twinx()
            axes[1].plot(yrs8, D8, marker="v", color="purple", label="D [trái]")
            ax2t.plot(yrs8, AI8, marker="d", color="crimson", linestyle="-.", label="AI [phải]")
            axes[1].set_title("Hạ tầng số D & AI Score"); axes[1].grid(True, alpha=0.4)
            axes[2].plot(yrs8, H8, marker="h", color="teal")
            axes[2].set_title("Tỷ lệ nhân lực số (H)"); axes[2].grid(True, alpha=0.4)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    except ImportError:
        st.error("Cần cài đặt: pip install cvxpy")


# ══════════════════════════════════════════════════════════════════
# BÀI 9 — LAO ĐỘNG & AI
# ══════════════════════════════════════════════════════════════════
elif choice == "👥 Bài 9 — Lao động & AI":
    st.header("Bài 9. Tác động AI tới thị trường lao động — Mô phỏng NetJob")
    st.markdown("""
    **Mô hình tối ưu CVXPY:** Phân bổ ngân sách AI (x_AI) và Nhân lực (x_H) để tối đa hoá
    **NetJob = Việc làm mới + Nâng cấp − Bị thay thế**, đồng thời đảm bảo an sinh lao động.
    """)

    try:
        import cvxpy as cp

        @st.cache_data
        def calc_bai9():
            sectors = ["Nông-Lâm","CN chế tạo","Xây dựng","Bán lẻ","Tài chính","Logistics","CNTT","Giáo dục"]
            L_jobs = np.array([13.20,11.50,4.80,7.80,0.55,1.95,0.62,2.15])*1e6
            risk   = np.array([18,42,25,38,52,35,28,22])/100
            a1 = np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
            b1 = np.array([45.0,28.0,35.0,32.0,22.0,30.0,20.0,55.0])
            c1 = np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
            d1 = np.array([50.0,32.0,42.0,38.0,26.0,36.0,24.0,62.0])
            N  = len(sectors)

            xAI = cp.Variable(N, nonneg=True)
            xH  = cp.Variable(N, nonneg=True)
            NewJob   = cp.multiply(a1, xAI)
            Upgrade  = cp.multiply(b1, xH)
            Displaced= cp.multiply(cp.multiply(c1, risk), xAI)
            Retrain  = cp.multiply(d1, xH)
            NetJob   = NewJob + Upgrade - Displaced

            cons = [cp.sum(xAI+xH)<=30000, NetJob>=0, Displaced<=Retrain]
            prob = cp.Problem(cp.Maximize(cp.sum(NetJob)), cons)
            prob.solve()

            df = pd.DataFrame({
                "Ngành": sectors,
                "x_AI (tỷ)": np.round(xAI.value,2),
                "x_H (tỷ)": np.round(xH.value,2),
                "Việc mới": np.round(NewJob.value,0),
                "Nâng cấp": np.round(Upgrade.value,0),
                "Bị thay thế": np.round(Displaced.value,0),
                "NetJob": np.round(NetJob.value,0),
            })
            return df, sectors, Displaced.value, Upgrade.value, L_jobs

        df9, sectors9, displaced9, upgrade9, Ljobs9 = calc_bai9()
        total_net = df9["NetJob"].sum()
        st.metric("Tổng NetJob ròng toàn hệ thống", f"{total_net:,.0f} việc làm")

        tab1, tab2, tab3 = st.tabs(["📊 Bảng kết quả", "📈 Biểu đồ", "🌊 Sankey Diagram"])

        with tab1:
            st.dataframe(df9.style.format({c:"{:,.0f}" for c in df9.columns if c!="Ngành"}),
                         use_container_width=True)

        with tab2:
            fig, axes = plt.subplots(1,2,figsize=(14,4))
            axes[0].barh(df9["Ngành"], df9["x_AI (tỷ)"], color="#4c72b0", label="x_AI")
            axes[0].barh(df9["Ngành"], df9["x_H (tỷ)"],  left=df9["x_AI (tỷ)"],
                         color="#dd8452", label="x_H")
            axes[0].set_title("Phân bổ ngân sách AI & Nhân lực"); axes[0].legend()
            axes[0].grid(axis="x", alpha=0.4)

            colors9 = ["#2ca02c" if v>=0 else "#d62728" for v in df9["NetJob"]]
            axes[1].barh(df9["Ngành"], df9["NetJob"], color=colors9)
            axes[1].axvline(0, color="black", linewidth=0.8)
            axes[1].set_title("NetJob ròng theo ngành"); axes[1].grid(axis="x", alpha=0.4)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with tab3:
            idx3 = [0, 2, 3]
            sources = [0,0,0,1,1,1,2,2,2]
            targets = [3,4,5,3,4,5,3,4,5]
            disp3  = displaced9[idx3]
            upgr3  = upgrade9[idx3]
            ret3   = Ljobs9[idx3] - disp3
            values = [disp3[0],upgr3[0],ret3[0], disp3[1],upgr3[1],ret3[1], disp3[2],upgr3[2],ret3[2]]
            labels_sk = [sectors9[i] for i in idx3] + ["Bị thay thế","Nâng cấp","Giữ nguyên"]
            fig_s = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, label=labels_sk,
                          color=["#4c72b0","#dd8452","#55a868","#d62728","#2ca02c","#9467bd"]),
                link=dict(source=sources, target=targets, value=values),
            )])
            fig_s.update_layout(title="Luồng dịch chuyển lao động — Ngành 1, 3, 4", height=450)
            st.plotly_chart(fig_s, use_container_width=True)

    except ImportError:
        st.error("Cần cài đặt: pip install cvxpy")


# ══════════════════════════════════════════════════════════════════
# BÀI 10 — STOCHASTIC SP
# ══════════════════════════════════════════════════════════════════
elif choice == "🎲 Bài 10 — Stochastic SP":
    st.header("Bài 10. Quy hoạch ngẫu nhiên hai giai đoạn (Stochastic SP)")
    st.markdown("""
    **Mô hình SP (Pyomo/HiGHS):** Tối đa hoá GDP kỳ vọng qua 4 kịch bản kinh tế toàn cầu.
    Tính **VSS** (giá trị thông tin ngẫu nhiên) và **EVPI** (giá trị thông tin hoàn hảo).
    """)

    try:
        import pyomo.environ as pyo

        @st.cache_data
        def calc_bai10():
            p   = {'s1':0.30,'s2':0.45,'s3':0.20,'s4':0.05}
            b1  = {'I':1.00,'D':1.10,'AI':1.25,'H':0.95}
            bs  = {
                ('s1','I'):1.25,('s1','D'):1.35,('s1','AI'):1.55,('s1','H'):1.05,
                ('s2','I'):1.00,('s2','D'):1.10,('s2','AI'):1.25,('s2','H'):0.95,
                ('s3','I'):0.75,('s3','D'):0.85,('s3','AI'):0.90,('s3','H'):1.00,
                ('s4','I'):0.40,('s4','D'):0.50,('s4','AI'):0.55,('s4','H'):1.10,
            }
            J=['I','D','AI','H']; S=list(p.keys())
            solver = pyo.SolverFactory('appsi_highs')

            def build(fix_x=None):
                m = pyo.ConcreteModel()
                m.J=pyo.Set(initialize=J); m.S=pyo.Set(initialize=S)
                m.p=pyo.Param(m.S,initialize=p); m.b=pyo.Param(m.J,initialize=b1)
                m.bs=pyo.Param(m.S,m.J,initialize=bs)
                m.x=pyo.Var(m.J,within=pyo.NonNegativeReals)
                m.y=pyo.Var(m.S,m.J,within=pyo.NonNegativeReals)
                if fix_x:
                    for j in J: m.x[j].fix(fix_x[j])
                m.b1c=pyo.Constraint(expr=sum(m.x[j] for j in J)<=65000)
                m.b2c=pyo.Constraint(m.S, rule=lambda mm,s: sum(mm.y[s,j] for j in J)<=15000)
                m.aic=pyo.Constraint(m.S, rule=lambda mm,s: mm.y[s,'AI']<=0.5*mm.x['H'])
                m.obj=pyo.Objective(
                    expr=sum(m.b[j]*m.x[j] for j in J)+sum(m.p[s]*sum(m.bs[s,j]*m.y[s,j] for j in J) for s in S),
                    sense=pyo.maximize)
                return m

            sp=build(); solver.solve(sp)
            SP_val=pyo.value(sp.obj)
            x_sp={j:pyo.value(sp.x[j]) for j in J}

            exp_b={j:sum(p[s]*bs[s,j] for s in S) for j in J}
            ev=pyo.ConcreteModel()
            ev.J=pyo.Set(initialize=J)
            ev.x=pyo.Var(ev.J,within=pyo.NonNegativeReals)
            ev.y=pyo.Var(ev.J,within=pyo.NonNegativeReals)
            ev.b1=pyo.Constraint(expr=sum(ev.x[j] for j in J)<=65000)
            ev.b2=pyo.Constraint(expr=sum(ev.y[j] for j in J)<=15000)
            ev.ai=pyo.Constraint(expr=ev.y['AI']<=0.5*ev.x['H'])
            ev.obj=pyo.Objective(expr=sum(b1[j]*ev.x[j] for j in J)+sum(exp_b[j]*ev.y[j] for j in J),sense=pyo.maximize)
            solver.solve(ev)
            x_ev={j:pyo.value(ev.x[j]) for j in J}

            eev_m=build(fix_x=x_ev); solver.solve(eev_m)
            EEV=pyo.value(eev_m.obj)
            VSS=SP_val-EEV

            WS=0
            for s in S:
                wm=build(); 
                for ss in S:
                    wm.y[ss,'AI'].fix(0) if ss!=s else None
                wm2=pyo.ConcreteModel()
                wm2.J=pyo.Set(initialize=J)
                wm2.x=pyo.Var(wm2.J,within=pyo.NonNegativeReals)
                wm2.y=pyo.Var(wm2.J,within=pyo.NonNegativeReals)
                wm2.b1=pyo.Constraint(expr=sum(wm2.x[j] for j in J)<=65000)
                wm2.b2=pyo.Constraint(expr=sum(wm2.y[j] for j in J)<=15000)
                wm2.ai=pyo.Constraint(expr=wm2.y['AI']<=0.5*wm2.x['H'])
                wm2.obj=pyo.Objective(expr=sum(b1[j]*wm2.x[j] for j in J)+sum(bs[s,j]*wm2.y[j] for j in J),sense=pyo.maximize)
                solver.solve(wm2)
                WS+=p[s]*pyo.value(wm2.obj)
            EVPI=WS-SP_val

            return x_sp, x_ev, SP_val, EEV, VSS, WS, EVPI, J, S, p, bs, b1

        with st.spinner("Đang giải mô hình Stochastic SP (Pyomo/HiGHS)…"):
            x_sp10,x_ev10,SP10,EEV10,VSS10,WS10,EVPI10,J10,S10,p10,bs10,b1_10 = calc_bai10()

        tab1, tab2, tab3 = st.tabs(["✅ Kết quả SP", "📐 VSS", "💎 EVPI"])

        with tab1:
            c1, c2 = st.columns(2)
            c1.markdown("**Quyết định x (SP — Here-and-Now):**")
            for j in J10: c1.metric(f"x[{j}]", f"{x_sp10[j]:,.0f} tỷ")
            c2.metric("GDP kỳ vọng tối ưu (SP)", f"{SP10:,.2f} tỷ")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.bar(J10, [x_sp10[j] for j in J10], color=["#4c72b0","#dd8452","#55a868","#c44e52"])
            ax.set_title("Quyết định First-stage tối ưu (SP)"); ax.set_ylabel("Tỷ VND")
            ax.grid(axis="y", alpha=0.4); st.pyplot(fig); plt.close(fig)

        with tab2:
            c1, c2, c3 = st.columns(3)
            c1.metric("SP (Stochastic)",    f"{SP10:,.2f} tỷ")
            c2.metric("EEV (Dùng trung bình)", f"{EEV10:,.2f} tỷ")
            c3.metric("VSS = SP − EEV",     f"{VSS10:,.2f} tỷ", delta="Giá trị tư duy xác suất")
            st.info("**VSS:** Số tỷ VND GDP bị mất đi nếu Chính phủ chỉ dùng 1 kịch bản trung bình thay vì lập kế hoạch xác suất.")

        with tab3:
            c1, c2, c3 = st.columns(3)
            c1.metric("WS (Wait-and-See)",  f"{WS10:,.2f} tỷ")
            c2.metric("SP",                 f"{SP10:,.2f} tỷ")
            c3.metric("EVPI = WS − SP",     f"{EVPI10:,.2f} tỷ", delta="Giá trị thông tin hoàn hảo")
            st.info("**EVPI:** Mức tối đa Chính phủ sẵn sàng trả để biết trước chính xác kịch bản kinh tế thế giới.")

    except ImportError:
        st.error("Cần cài đặt: pip install pyomo && conda install -c conda-forge glpk (hoặc HiGHS)")
    except Exception as e:
        st.error(f"Lỗi khi chạy Pyomo: {e}")


# ══════════════════════════════════════════════════════════════════
# BÀI 11 — Q-LEARNING RL
# ══════════════════════════════════════════════════════════════════
elif choice == "🤖 Bài 11 — Q-learning RL":
    st.header("Bài 11. Học tăng cường (Q-learning) — Chính sách phân bổ ngân sách")
    st.markdown("""
    **Môi trường Gymnasium tùy chỉnh** mô phỏng nền kinh tế Việt Nam (10 bước = 10 năm).
    **Q-learning** học chính sách phân bổ ngân sách tối ưu qua 10.000 episode.
    """)

    try:
        import gymnasium as gym
        from gymnasium import spaces

        class VNEconEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.action_space      = spaces.Discrete(5)
                self.observation_space = spaces.MultiDiscrete([3,3,3,3])
                self.T = 10
                self.alloc = {
                    0: np.array([0.70,0.10,0.10,0.10]),
                    1: np.array([0.40,0.25,0.15,0.20]),
                    2: np.array([0.25,0.45,0.15,0.15]),
                    3: np.array([0.20,0.20,0.45,0.15]),
                    4: np.array([0.30,0.20,0.10,0.40]),
                }
                self.w = np.array([0.40,0.25,0.20,0.15])

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.state = np.array([1,1,0,1]); self.t = 0
                self.K,self.D,self.AI,self.H = 27500.0,20.3,86.0,30.0
                self.Y = self.K**0.33*(54.0)**0.42*self.D**0.10*self.AI**0.08*self.H**0.07
                return self.state, {}

            def step(self, action):
                a = self.alloc[action]; budget = 1000; prev_Y = self.Y
                self.K  += a[0]*budget; self.D  += a[1]*budget/100
                self.AI += a[2]*budget/20; self.H += a[3]*budget/200
                self.Y = self.K**0.33*(54.0)**0.42*self.D**0.10*self.AI**0.08*self.H**0.07
                dY = self.Y - prev_Y
                ur = max(0, 40-self.H); cr = self.AI*0.1; em = self.K*0.05
                reward = self.w[0]*dY - self.w[1]*ur - self.w[2]*cr - self.w[3]*em
                self.t += 1; done = self.t >= self.T
                s0 = min(2,max(0,int(dY/50))); s1 = min(2,max(0,int(self.D/30)))
                s2 = min(2,max(0,int(self.AI/50))); s3 = min(2,max(0,int(ur/10)))
                self.state = np.array([s0,s1,s2,s3])
                return self.state, reward, done, False, {}

        @st.cache_data
        def train_ql(episodes=8000):
            env = VNEconEnv()
            Q   = np.zeros((3,3,3,3,5))
            alpha, gamma = 0.1, 0.95
            hist = []
            for ep in range(episodes):
                s,_ = env.reset(); tot = 0
                while True:
                    eps = max(0.05, 1.0 - ep/4000)
                    a   = env.action_space.sample() if np.random.rand()<eps else int(np.argmax(Q[tuple(s)]))
                    s2,r,done,_,_ = env.step(a)
                    Q[tuple(s)+(a,)] += alpha*(r+gamma*np.max(Q[tuple(s2)])-Q[tuple(s)+(a,)])
                    s = s2; tot += r
                    if done: break
                hist.append(tot)
            return Q, hist

        col_train, _ = st.columns([2,1])
        with col_train:
            n_ep = st.slider("Số episode huấn luyện", 2000, 10000, 6000, step=1000)

        with st.spinner(f"Đang huấn luyện Q-learning ({n_ep} episodes)…"):
            Q11, hist11 = train_ql(n_ep)

        # Đánh giá
        def evaluate(policy, n=30):
            env = VNEconEnv(); rews = []
            for _ in range(n):
                s,_ = env.reset(); er=0
                while True:
                    if policy=="learned": a=int(np.argmax(Q11[tuple(s)]))
                    elif policy=="balanced": a=1
                    elif policy=="ai_lead": a=3
                    else: a=env.action_space.sample()
                    s,r,done,_,_=env.step(a); er+=r
                    if done: break
                rews.append(er)
            return np.mean(rews)

        tab1, tab2, tab3 = st.tabs(["📈 Đường học", "🏅 So sánh chính sách", "🧠 Q-table heatmap"])

        with tab1:
            window = 200
            smooth = pd.Series(hist11).rolling(window).mean()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hist11, alpha=0.2, color="#4c72b0", linewidth=0.6, label="Raw reward")
            ax.plot(smooth, color="#e63946", linewidth=2,              label=f"MA-{window}")
            ax.set_xlabel("Episode"); ax.set_ylabel("Tổng phần thưởng")
            ax.set_title("Đường học của Q-learning"); ax.legend(); ax.grid(True, alpha=0.4)
            st.pyplot(fig); plt.close(fig)

        with tab2:
            policies = {
                "Q-learning (Learned)": evaluate("learned"),
                "Cân bằng (a=1)":       evaluate("balanced"),
                "AI dẫn dắt (a=3)":     evaluate("ai_lead"),
                "Ngẫu nhiên":           evaluate("random"),
            }
            df_pol = pd.DataFrame({"Chính sách": list(policies.keys()),
                                   "Phần thưởng tb": list(policies.values())})
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(df_pol.style.format({"Phần thưởng tb": "{:,.2f}"}), use_container_width=True)
            with c2:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                cols_bar = ["#2ca02c" if i==0 else "#9467bd" for i in range(4)]
                ax.bar(df_pol["Chính sách"], df_pol["Phần thưởng tb"], color=cols_bar)
                ax.set_ylabel("Phần thưởng tích lũy trung bình")
                ax.set_title("So sánh hiệu quả các chính sách phân bổ")
                ax.grid(axis="y", alpha=0.4); plt.xticks(rotation=15)
                st.pyplot(fig); plt.close(fig)

        with tab3:
            st.markdown("Q-table tại trạng thái ban đầu `[1,1,0,1]` (GDP medium, Digital medium, AI low, U-risk medium):")
            init_q = Q11[1,1,0,1]
            action_names = ["Truyền thống","Cân bằng","Số hóa nhanh","AI dẫn dắt","Bao trùm"]
            df_qt = pd.DataFrame({"Hành động": action_names, "Q-value": init_q})
            best_a = int(np.argmax(init_q))
            st.dataframe(df_qt.style.format({"Q-value":"{:.4f}"}), use_container_width=True)
            st.success(f"✅ Hành động tối ưu tại trạng thái ban đầu: **{action_names[best_a]}**")

    except ImportError:
        st.error("Cần cài đặt: pip install gymnasium")


# ══════════════════════════════════════════════════════════════════
# BÀI 12 — AIDEOM TÍCH HỢP
# ══════════════════════════════════════════════════════════════════
elif choice == "🇻🇳 Bài 12 — AIDEOM tích hợp":
    try:
        import pulp

        # ── Dữ liệu chung ──
        alpha_m, beta_m, gamma_m, delta_m, theta_m = 0.33, 0.42, 0.10, 0.08, 0.07
        K_25, L_25, D_25, AI_25, H_25 = 25900, 53.4, 19.5, 80.1, 29.2
        A_bar = 0.98
        sectors12 = ["Nông-Lâm","CN chế tạo","Xây dựng","Bán lẻ","Tài chính","Logistics","CNTT","Giáo dục"]
        L_jobs12  = np.array([13.20,11.50,4.80,7.80,0.55,1.95,0.62,2.15])*1e6
        risk12    = np.array([18,42,25,38,52,35,28,22])/100
        a1_12 = np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
        b1_12 = np.array([45.0,28.0,35.0,32.0,22.0,30.0,20.0,55.0])
        c1_12 = np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])

        beta_m4 = np.array([
            [1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],[1.05,0.95,0.85,1.15],
            [1.20,0.75,0.45,1.35],[0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25],
        ])

        @st.cache_data
        def opt_s5(B):
            prob = pulp.LpProblem("S5", pulp.LpMaximize)
            x = pulp.LpVariable.dicts("x",((r,j) for r in range(6) for j in range(4)),lowBound=0)
            prob += pulp.lpSum(beta_m4[r,j]*x[r,j] for r in range(6) for j in range(4))
            prob += pulp.lpSum(x[r,j] for r in range(6) for j in range(4)) <= B
            for r in range(6):
                prob += pulp.lpSum(x[r,j] for j in range(4)) >= B*0.05
                prob += pulp.lpSum(x[r,j] for j in range(4)) <= B*0.30
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            item_totals = [sum(x[r,j].varValue for r in range(6)) for j in range(4)]
            s = sum(item_totals)
            return [v/s for v in item_totals] if s>0 else [0.25]*4

        def forecast_gdp(ratios, B):
            alloc = [r*B for r in ratios]
            K2030  = K_25 + alloc[0]*0.5
            L2030  = L_25*(1.01**5)
            D2030  = D_25 + alloc[1]*0.001
            AI2030 = AI_25+ alloc[2]*0.005
            H2030  = H_25 + alloc[3]*0.002
            A2030  = A_bar*(1.012**5)
            gdp = A2030*(K2030**alpha_m)*(L2030**beta_m)*(D2030**gamma_m)*(AI2030**delta_m)*(H2030**theta_m)
            return gdp, alloc

        def calc_labor(ai_b, h_b):
            xAI = np.ones(8)*(ai_b/8); xH = np.ones(8)*(h_b/8)
            disp = c1_12*risk12*xAI
            upgr = b1_12*xH
            net  = a1_12*xAI + b1_12*xH - disp
            return disp, upgr, net

        # ── Sidebar kịch bản ──
        st.title("AIDEOM-VN: Hệ thống Hỗ trợ Ra Quyết định Tích hợp")
        st.sidebar.markdown("---")
        st.sidebar.header("🎛️ Kịch bản chính sách")
        SCENARIOS = {
            "S1. Truyền thống":  [0.70,0.10,0.10,0.10],
            "S2. Số hóa nhanh":  [0.25,0.45,0.15,0.15],
            "S3. AI dẫn dắt":    [0.20,0.20,0.45,0.15],
            "S4. Bao trùm số":   [0.30,0.20,0.10,0.40],
            "S5. Tối ưu (LP)":   "OPTIMIZE",
        }
        sc_name = st.sidebar.selectbox("Kịch bản phân bổ 2026-2030:", list(SCENARIOS.keys()))
        total_B = st.sidebar.number_input("Tổng ngân sách (tỷ VND):", 50000, 300000, 100000, 10000)

        ratios12 = opt_s5(total_B) if SCENARIOS[sc_name]=="OPTIMIZE" else SCENARIOS[sc_name]
        GDP12, alloc12 = forecast_gdp(ratios12, total_B)
        disp12, upgr12, net12 = calc_labor(alloc12[2], alloc12[3])

        # ── KPI ──
        ka, kb, kc, kd = st.columns(4)
        ka.metric("Dự báo GDP 2030 (nghìn tỷ)",   f"{GDP12:,.0f}")
        kb.metric("Ngân sách AI+Nhân lực (tỷ)",    f"{alloc12[2]+alloc12[3]:,.0f}")
        kc.metric("NetJob ròng toàn hệ thống",      f"{np.sum(net12):,.0f}")
        kd.metric("Kịch bản đang chọn", sc_name[:3])

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng quan", "💰 Phân bổ ngân sách", "👥 Tác động lao động", "🗺️ Vùng miền"])

        with tab1:
            df_kpi = pd.DataFrame({
                "Hạng mục": ["Vốn (K)","Hạ tầng số (D)","AI","Nhân lực (H)"],
                "Ngân sách (tỷ)": alloc12,
                "Tỷ trọng (%)": [r*100 for r in ratios12],
            })
            c1, c2 = st.columns([1,2])
            with c1:
                st.dataframe(df_kpi.style.format({"Ngân sách (tỷ)":"{:,.0f}","Tỷ trọng (%)" :"{:.1f}"}),
                             use_container_width=True)
            with c2:
                fig_kpi = px.bar(df_kpi, x="Hạng mục", y="Ngân sách (tỷ)",
                                 color="Hạng mục", title=f"Phân bổ ngân sách — {sc_name}")
                st.plotly_chart(fig_kpi, use_container_width=True)

        with tab2:
            fig_pie = px.pie(
                names=["Vốn (K)","Hạ tầng số (D)","AI","Nhân lực (H)"],
                values=ratios12, hole=0.45,
                title=f"Cơ cấu phân bổ ngân sách — {sc_name}",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with tab3:
            df_lab = pd.DataFrame({
                "Ngành": sectors12,
                "Bị thay thế": np.round(disp12,0),
                "Nâng cấp":    np.round(upgr12,0),
                "NetJob":      np.round(net12,0),
            })
            c1, c2 = st.columns([1,2])
            with c1:
                st.dataframe(df_lab.style.format({c:"{:,.0f}" for c in df_lab.columns if c!="Ngành"}),
                             use_container_width=True)
            with c2:
                src = [0,0,0,1,1,1,2,2,2]
                tgt = [3,4,5,3,4,5,3,4,5]
                idx = [1,3,4]
                vals = [disp12[i] for i in idx] + [upgr12[i] for i in idx] + [max(0,net12[i]) for i in idx]
                vals_sk = []
                for i in range(3):
                    vals_sk += [disp12[idx[i]], upgr12[idx[i]], max(0,net12[idx[i]])]
                fig_s = go.Figure(data=[go.Sankey(
                    node=dict(
                        label=[sectors12[i] for i in idx]+["Bị thay thế","Nâng cấp","Việc làm mới"],
                        color=["#4c72b0","#dd8452","#55a868","#d62728","#2ca02c","#9467bd"],
                    ),
                    link=dict(source=src, target=tgt, value=vals_sk),
                )])
                fig_s.update_layout(title="Luồng dịch chuyển lao động", height=380)
                st.plotly_chart(fig_s, use_container_width=True)

        with tab4:
            regions12 = ['Trung du MN Bắc','ĐB sông Hồng','BTBộ & DHMT','Tây Nguyên','Đông Nam Bộ','ĐB SCL']
            reg_alloc = [ratios12[0]*total_B*w for w in [0.10,0.28,0.17,0.08,0.25,0.12]]
            fig_reg = px.bar(
                x=regions12, y=reg_alloc,
                labels={"x":"Vùng","y":"Ngân sách (tỷ VND)"},
                title="Phân bổ ngân sách ước tính theo vùng kinh tế",
                color=regions12, color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_reg, use_container_width=True)

    except ImportError:
        st.error("Cần cài đặt: pip install pulp plotly")
