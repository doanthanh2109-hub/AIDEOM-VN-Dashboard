import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pulp

# ==========================================
# CẤU HÌNH TRANG
# ==========================================
st.set_page_config(page_title="VN AIDEOM-VN", layout="wide", page_icon="🇻🇳")

# CSS tùy chỉnh màu sắc
st.markdown("""
<style>
div[data-testid="metric-container"] > div > div > div { color: #ff4b4b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CÁC HÀM RENDER TỪNG BÀI
# ==========================================

# --- LOGIC BÀI 1 (Đã tích hợp) ---
@st.cache_data
def calculate_bai1_data():
    Year = np.array([2020, 2021, 2022, 2023, 2024, 2025])
    Y = np.array([8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6])
    K = np.array([16500, 17800, 19600, 21300, 23500, 25900])
    L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
    D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
    AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
    H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])
    alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
    data = pd.DataFrame({"Year": Year, "Y": Y, "K": K, "L": L, "D": D, "AI": AI, "H": H})
    data["A"] = Y / ((K ** alpha) * (L ** beta) * (D ** gamma) * (AI ** delta) * (H ** theta))
    return data

def render_bai1():
    st.header("📈 Bài 1 — Cobb-Douglas + AI")
    data = calculate_bai1_data()
    st.dataframe(data)
    fig, ax = plt.subplots()
    ax.plot(data["Year"], data["A"], marker="o")
    ax.set_title("Total Factor Productivity (TFP)")
    st.pyplot(fig)
    plt.close(fig)

# --- KHUNG RENDER CÁC BÀI KHÁC (Đợi bạn điền code) ---
def render_bai2():
    st.header("💰 Bài 2 — LP ngân sách số")
    st.info("Đang chờ cập nhật code bài 2...")

def render_bai12():
    # Giữ nguyên code Bài 12 cũ của bạn ở đây...
    st.header("🇻🇳 Bài 12 — AIDEOM tích hợp")
    # ... (code bài 12)

# ==========================================
# ĐIỀU HƯỚNG TRANG
# ==========================================
menu = {
    "🏠 Trang chủ": None,
    "📈 Bài 1 — Cobb-Douglas + AI": render_bai1,
    "💰 Bài 2 — LP ngân sách số": render_bai2,
    "🇻🇳 Bài 12 — AIDEOM tích hợp": render_bai12
}

choice = st.sidebar.radio("Điều hướng", list(menu.keys()))

if choice == "🏠 Trang chủ":
    st.title("VN AIDEOM-VN")
    # ... (code hiển thị KPI trang chủ cũ của bạn)
elif menu[choice] is not None:
    menu[choice]()
