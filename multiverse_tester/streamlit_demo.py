#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интерактивное веб-демо MultiverseTester — «пузырь жизни».

Показывает, как меняется пригодность вселенной для жизни при изменении
фундаментальных констант. Запуск: streamlit run -m multiverse_tester.streamlit_demo
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(
    page_title="MultiverseTester — Пузырь жизни",
    page_icon="🌌",
    layout="wide",
)

# Загружаем движок
@st.cache_resource
def load_engine():
    from multiverse_tester import UniverseParameters, UniverseAnalyzer, UniversalConstants
    return UniverseParameters, UniverseAnalyzer, UniversalConstants

UniverseParameters, UniverseAnalyzer, UniversalConstants = load_engine()
const = UniversalConstants()

# === Стиль ===
st.markdown("""
<style>
    .stProgress > div > div > div { background: linear-gradient(90deg, #c0392b, #27ae60); }
    .metric-card { 
        padding: 1rem; 
        border-radius: 0.5rem; 
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        margin: 0.5rem 0;
    }
    .score-high { color: #27ae60; }
    .score-mid { color: #f39c12; }
    .score-low { color: #c0392b; }
</style>
""", unsafe_allow_html=True)

st.title("🌌 MultiverseTester — Пузырь жизни")
st.markdown("*Исследуйте, как фундаментальные константы влияют на пригодность вселенной для жизни*")

# === Боковая панель: слайдеры ===
st.sidebar.header("🔬 Параметры вселенной")
st.sidebar.markdown("Относительно нашей Вселенной (1.0 = наше значение)")

fix_e = st.sidebar.checkbox(
    "Фиксировать заряд e (ε₀ и ħ влияют на α)",
    value=False,
    help="При включении α = e²/(4π ε₀ ℏ c) — меняя ε₀ или ħ, вы меняете α",
)

if fix_e:
    alpha_val = None  # будет вычислено из e, ε₀, ħ
else:
    alpha_val = st.sidebar.slider(
        "α (постоянная тонкой структуры)",
        min_value=0.003,
        max_value=0.05,
        value=1/137.036,
        step=0.0005,
        format="%.4f",
        help="1/137 ≈ наша Вселенная",
    )

m_p_ratio = st.sidebar.slider("m_p / m_p₀ (масса протона)", 0.3, 3.0, 1.0, 0.1)
m_e_ratio = st.sidebar.slider("m_e / m_e₀ (масса электрона)", 0.3, 3.0, 1.0, 0.1)
G_ratio = st.sidebar.slider("G / G₀ (гравитационная постоянная)", 0.2, 5.0, 1.0, 0.1)
c_ratio = st.sidebar.slider("c / c₀ (скорость света)", 0.5, 2.0, 1.0, 0.1)
hbar_ratio = st.sidebar.slider("ℏ / ℏ₀ (постоянная Планка)", 0.5, 2.0, 1.0, 0.1)
eps_ratio = st.sidebar.slider("ε₀ / ε₀₀ (диэлектрическая проницаемость)", 0.3, 3.0, 1.0, 0.1)

st.sidebar.markdown("---")
show_landscape = st.sidebar.checkbox("Показать ландшафт (α vs m_p)", value=True)
landscape_res = st.sidebar.slider("Разрешение ландшафта", 15, 40, 25, help="Больше = точнее, но медленнее")

# === Расчёт пригодности ===
try:
    if fix_e:
        u = UniverseParameters(
            name="Custom",
            e=const.e,
            fix_e=True,
            m_p=m_p_ratio * const.m_p,
            m_e=m_e_ratio * const.m_e,
            G=G_ratio * const.G,
            c=c_ratio * const.c,
            hbar=hbar_ratio * const.hbar,
            epsilon_0=eps_ratio * const.epsilon_0,
        )
        alpha_val = u.alpha  # для отображения
    else:
        u = UniverseParameters(
            name="Custom",
            alpha=alpha_val,
            m_p=m_p_ratio * const.m_p,
            m_e=m_e_ratio * const.m_e,
            G=G_ratio * const.G,
            c=c_ratio * const.c,
            hbar=hbar_ratio * const.hbar,
            epsilon_0=eps_ratio * const.epsilon_0,
        )
    analyzer = UniverseAnalyzer(u)
    index, score, metrics = analyzer.calculate_habitability_index()
    error_msg = None
except Exception as e:
    index, score, metrics = None, 0.0, {}
    error_msg = str(e)

# === Основная область ===
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📊 Индекс пригодности")
    if error_msg:
        st.error(f"Ошибка: {error_msg}")
    else:
        if score > 0.8:
            st.success(f"**{score:.2%}** — Оптимально")
        elif score > 0.6:
            st.info(f"**{score:.2%}** — Пригодно")
        elif score > 0.3:
            st.warning(f"**{score:.2%}** — Маргинально")
        else:
            st.error(f"**{score:.2%}** — Непригодно")
        
        st.progress(score)

with col2:
    st.subheader("📈 Метрики")
    if not error_msg and metrics:
        for k, v in sorted(metrics.items()):
            pct = v * 100
            st.caption(f"**{k}**: {pct:.0f}%")
            st.progress(v)

with col3:
    st.subheader("📍 Позиция")
    st.markdown(f"α = {alpha_val:.6f}")
    st.markdown(f"m_p/m_p₀ = {m_p_ratio:.2f}")
    st.markdown(f"m_e/m_e₀ = {m_e_ratio:.2f}")
    if fix_e:
        st.caption("α из e, ε₀, ħ")
    st.markdown(f"ℏ/ℏ₀ = {hbar_ratio:.2f}, ε₀/ε₀₀ = {eps_ratio:.2f}")

# === Пузырь жизни: 2D ландшафт ===
if show_landscape and not error_msg:
    st.markdown("---")
    st.subheader("🗺️ Пузырь жизни: ландшафт (α, m_p)")
    st.caption("Зелёный = пригодно, красный = непригодно. Белый крест — ваша текущая позиция.")
    
    @st.cache_data(show_spinner="Строим ландшафт...")
    def compute_landscape(n_alpha: int, n_mp: int, m_e_r: float, G_r: float,
                          c_r: float, hbar_r: float, eps_r: float):
        alphas = np.linspace(1/300, 1/30, n_alpha)
        m_p_ratios = np.linspace(0.5, 2.0, n_mp)
        score_map = np.zeros((n_alpha, n_mp))
        for i, a in enumerate(alphas):
            for j, mp in enumerate(m_p_ratios):
                try:
                    u_ij = UniverseParameters(
                        alpha=a,
                        m_p=mp * const.m_p,
                        m_e=m_e_r * const.m_e,
                        G=G_r * const.G,
                        c=c_r * const.c,
                        hbar=hbar_r * const.hbar,
                        epsilon_0=eps_r * const.epsilon_0,
                    )
                    _, s, _ = UniverseAnalyzer(u_ij).calculate_habitability_index()
                    score_map[i, j] = s
                except Exception:
                    score_map[i, j] = 0
        return alphas, m_p_ratios, score_map
    
    alphas, m_p_arr, score_map = compute_landscape(
        landscape_res, landscape_res,
        m_e_ratio, G_ratio, c_ratio, hbar_ratio, eps_ratio
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        score_map.T,
        aspect='auto',
        extent=[alphas[0], alphas[-1], m_p_arr[0], m_p_arr[-1]],
        origin='lower',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel('α (постоянная тонкой структуры)')
    ax.set_ylabel('m_p / m_p₀')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Наша m_p')
    ax.axvline(1/137.036, color='gray', linestyle='--', alpha=0.5, label='Наша α')
    # Текущая позиция
    ax.plot(alpha_val, m_p_ratio, 'w+', markersize=20, markeredgewidth=3)
    ax.plot(alpha_val, m_p_ratio, 'k+', markersize=18, markeredgewidth=1)
    plt.colorbar(im, ax=ax, label='Пригодность')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close()

st.markdown("---")
st.caption("MultiverseTester • Симуляция пригодности вселенных для жизни")
