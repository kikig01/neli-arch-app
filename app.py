import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="Анализ на полукръгла арка",
    layout="wide"
)

st.title("Интерактивен анализ на полукръгла арка")
st.caption(
    "Модел: полукръгла запъната арка. "
    "Показва геометрия, линия на натиска, M(θ), N(θ), e = M/N и проверка на ядрото."
)


# ============================================================
# Inputs
# ============================================================

st.sidebar.header("Входни данни")

R = st.sidebar.number_input(
    "Радиус R [m]",
    min_value=0.10,
    value=5.00,
    step=0.50
)

L = 2 * R

st.sidebar.write(f"Отвор L = 2R = {L:.2f} m")

h = st.sidebar.number_input(
    "Дебелина h [m]",
    min_value=0.01,
    value=0.80,
    step=0.05
)

q = st.sidebar.number_input(
    "Натоварване q [kN/m]",
    min_value=0.00,
    value=20.00,
    step=1.00
)

dtheta_deg = st.sidebar.number_input(
    "Стъпка Δθ [градуси]",
    min_value=1.00,
    max_value=30.00,
    value=5.00,
    step=1.00
)

material = st.sidebar.selectbox(
    "Материал",
    ["Зидария", "Камък", "Бетон", "Тухла", "Друго"]
)

show_pressure_line = st.sidebar.checkbox(
    "Покажи линия на натиска върху арката",
    value=True
)

show_blocks = st.sidebar.checkbox(
    "Покажи деление на блокове",
    value=True
)

st.sidebar.info(
    "Ъгълът θ се мери от лявата опора към върха и после към дясната опора: "
    "θ = 0° при лявата опора, θ = 90° във върха, θ = 180° при дясната опора."
)


# ============================================================
# Geometry
# ============================================================

#R = L / 2
# R is now given directly as input.
# For a semicircular arch, L = 2R.

if h >= 2 * R:
    st.error("Дебелината h е твърде голяма спрямо радиуса R.")
    st.stop()

theta_deg = np.arange(0, 180 + dtheta_deg, dtheta_deg)
theta_deg = theta_deg[theta_deg <= 180]
theta = np.radians(theta_deg)

# Axis of arch
x_axis = -R * np.cos(theta)
y_axis = R * np.sin(theta)

# Intrados and extrados
R_intrados = R - h / 2
R_extrados = R + h / 2

x_intrados = -R_intrados * np.cos(theta)
y_intrados = R_intrados * np.sin(theta)

x_extrados = -R_extrados * np.cos(theta)
y_extrados = R_extrados * np.sin(theta)

s = R * theta


# ============================================================
# Structural formulas
# ============================================================

V_A = (q * np.pi * R) / 2
V_B = V_A

H_A = (q * R) / np.pi
H_B = H_A

M_A = -q * R**2 * (1 - 2 / np.pi)
M_B = M_A

M = q * R**2 * (
    2 / np.pi
    - np.cos(theta)
    - theta * np.sin(theta)
    + (1 / np.pi) * np.sin(theta)
)

N = -q * R * (
    np.cos(theta)
    + theta * np.sin(theta)
    + (1 / np.pi) * np.sin(theta)
)

e = np.divide(
    M,
    N,
    out=np.full_like(M, np.nan),
    where=np.abs(N) > 1e-9
)

kernel_plus = h / 6
kernel_minus = -h / 6

inside_kernel = np.abs(e) <= kernel_plus
critical_intrados = e > kernel_plus
critical_extrados = e < kernel_minus


# ============================================================
# Pressure line for visualisation
# ============================================================

# Radial direction
radial_x = -np.cos(theta)
radial_y = np.sin(theta)

# Real pressure line
x_pressure_real = x_axis + e * radial_x
y_pressure_real = y_axis + e * radial_y

# For the geometry plot only:
# If the pressure line is very far outside the arch, hide those points
# so the arch does not become tiny.
max_visual_offset = h / 2
e_visual = e.copy()
e_visual[np.abs(e_visual) > max_visual_offset] = np.nan

x_pressure_visual = x_axis + e_visual * radial_x
y_pressure_visual = y_axis + e_visual * radial_y


# ============================================================
# Table
# ============================================================

status = np.where(
    critical_intrados,
    "Критично към интрадоса",
    np.where(
        critical_extrados,
        "Критично към екстрадоса",
        "В ядрото"
    )
)

df = pd.DataFrame({
    "θ [deg]": theta_deg,
    "θ [rad]": theta,
    "s [m]": s,
    "x [m]": x_axis,
    "y [m]": y_axis,
    "M(θ) [kNm]": M,
    "N(θ) [kN]": N,
    "e = M/N [m]": e,
    "+h/6 [m]": np.full_like(theta, kernel_plus),
    "-h/6 [m]": np.full_like(theta, kernel_minus),
    "Проверка": status
})


# ============================================================
# Support reactions
# ============================================================

st.header("Опорни реакции")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("R [m]", f"{R:.3f}")
col2.metric("V_A [kN]", f"{V_A:.3f}")
col3.metric("V_B [kN]", f"{V_B:.3f}")
col4.metric("H_A = H_B [kN]", f"{H_A:.3f}")
col5.metric("M_A = M_B [kNm]", f"{M_A:.3f}")


# ============================================================
# Geometry plot
# ============================================================

st.header("Геометрия на арката")

fig_geom = go.Figure()

# Ground/support line
ground_extension = 0.18 * L
fig_geom.add_trace(go.Scatter(
    x=[-R - ground_extension, R + ground_extension],
    y=[0, 0],
    mode="lines",
    line=dict(color="red", width=2),
    name="Линия на опорите"
))

# Extrados
fig_geom.add_trace(go.Scatter(
    x=x_extrados,
    y=y_extrados,
    mode="lines",
    line=dict(width=3),
    name="Екстрадос"
))

# Intrados
fig_geom.add_trace(go.Scatter(
    x=x_intrados,
    y=y_intrados,
    mode="lines",
    line=dict(width=3),
    name="Интрадос"
))

# Axis
fig_geom.add_trace(go.Scatter(
    x=x_axis,
    y=y_axis,
    mode="lines",
    line=dict(width=2, dash="dash"),
    name="Ос на арката"
))

# Radial block divisions
if show_blocks:
    block_angles_deg = np.arange(0, 181, 10)
    block_angles = np.radians(block_angles_deg)

    for ang in block_angles:
        xi = -R_intrados * np.cos(ang)
        yi = R_intrados * np.sin(ang)

        xe = -R_extrados * np.cos(ang)
        ye = R_extrados * np.sin(ang)

        fig_geom.add_trace(go.Scatter(
            x=[xi, xe],
            y=[yi, ye],
            mode="lines",
            line=dict(color="red", width=1),
            showlegend=False,
            hoverinfo="skip"
        ))

# Supports
fig_geom.add_trace(go.Scatter(
    x=[-R, R],
    y=[0, 0],
    mode="markers",
    marker=dict(size=14),
    name="Опори"
))

# Pressure line visual, only if it lies within the thickness
if show_pressure_line:
    fig_geom.add_trace(go.Scatter(
        x=x_pressure_visual,
        y=y_pressure_visual,
        mode="lines+markers",
        line=dict(width=3),
        marker=dict(size=5),
        name="Линия на натиска в сечението"
    ))

# Span arrow
span_y = -0.16 * R
fig_geom.add_annotation(
    x=-R,
    y=span_y,
    ax=R,
    ay=span_y,
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=3,
    arrowsize=1,
    arrowwidth=2
)

fig_geom.add_annotation(
    x=R,
    y=span_y,
    ax=-R,
    ay=span_y,
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=3,
    arrowsize=1,
    arrowwidth=2
)

fig_geom.add_annotation(
    x=0,
    y=span_y - 0.04 * R,
    text="ОТВОР",
    showarrow=False,
    font=dict(size=16)
)

# Rise arrow
fig_geom.add_annotation(
    x=0,
    y=R,
    ax=0,
    ay=0,
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=3,
    arrowsize=1,
    arrowwidth=2
)

fig_geom.add_annotation(
    x=0,
    y=0,
    ax=0,
    ay=R,
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=3,
    arrowsize=1,
    arrowwidth=2
)

fig_geom.add_annotation(
    x=0.08 * R,
    y=0.5 * R,
    text="СТРЕЛА",
    showarrow=False,
    font=dict(size=16)
)

# Labels
fig_geom.add_annotation(
    x=0,
    y=R_extrados + 0.12 * R,
    text="КЛЮЧ",
    showarrow=True,
    ax=0.18 * R,
    ay=R_extrados + 0.35 * R,
    font=dict(size=16)
)

fig_geom.add_annotation(
    x=0.62 * R,
    y=0.76 * R_extrados,
    text="ЕКСТРАДОС",
    showarrow=True,
    ax=0.95 * R,
    ay=0.98 * R,
    font=dict(size=16)
)

fig_geom.add_annotation(
    x=-0.45 * R,
    y=0.52 * R_intrados,
    text="ИНТРАДОС",
    showarrow=True,
    ax=-0.68 * R,
    ay=0.68 * R,
    font=dict(size=16)
)

fig_geom.add_annotation(
    x=0.96 * R,
    y=0.62 * R,
    text="ДЕБЕЛИНА",
    showarrow=True,
    ax=1.25 * R,
    ay=0.78 * R,
    font=dict(size=16)
)

fig_geom.add_annotation(
    x=-R - 0.12 * R,
    y=-0.08 * R,
    text="ОПОРА",
    showarrow=True,
    ax=-R - 0.45 * R,
    ay=-0.22 * R,
    font=dict(size=16)
)

fig_geom.add_annotation(
    x=R + 0.12 * R,
    y=-0.08 * R,
    text="ОПОРА",
    showarrow=True,
    ax=R + 0.45 * R,
    ay=-0.22 * R,
    font=dict(size=16)
)

# Fixed axis limits so the arch stays readable
fig_geom.update_xaxes(
    range=[-R - 0.55 * R, R + 0.55 * R],
    title="x [m]",
    zeroline=False
)

fig_geom.update_yaxes(
    range=[-0.32 * R, R_extrados + 0.45 * R],
    title="y [m]",
    scaleanchor="x",
    scaleratio=1,
    zeroline=False
)

fig_geom.update_layout(
    title="Арка: интрадос, екстрадос, ос и линия на натиска",
    height=720,
    legend=dict(orientation="h", y=-0.15),
    margin=dict(l=30, r=30, t=80, b=80)
)

st.plotly_chart(fig_geom, use_container_width=True)


# ============================================================
# Warning if pressure line is outside
# ============================================================

if show_pressure_line and np.any(np.abs(e) > h / 2):
    st.warning(
        "Част от реалната линия на натиска е извън дебелината на арката. "
        "В геометричната схема са показани само точките, които попадат в дебелината, "
        "за да не се деформира мащабът на чертежа. Пълните стойности са в таблицата."
    )


# ============================================================
# Moment diagram
# ============================================================

st.header("Диаграма на огъващия момент M(θ)")

fig_M = go.Figure()

fig_M.add_trace(go.Scatter(
    x=theta_deg,
    y=M,
    mode="lines+markers",
    name="M(θ) [kNm]"
))

fig_M.add_hline(
    y=0,
    line_dash="dash"
)

fig_M.update_layout(
    title="Огъващ момент M(θ)",
    xaxis_title="θ [градуси]",
    yaxis_title="M(θ) [kNm]",
    height=450
)

st.plotly_chart(fig_M, use_container_width=True)


# ============================================================
# Normal force diagram
# ============================================================

st.header("Диаграма на нормалната сила N(θ)")

fig_N = go.Figure()

fig_N.add_trace(go.Scatter(
    x=theta_deg,
    y=N,
    mode="lines+markers",
    name="N(θ) [kN]"
))

fig_N.add_hline(
    y=0,
    line_dash="dash"
)

fig_N.update_layout(
    title="Нормална сила N(θ)",
    xaxis_title="θ [градуси]",
    yaxis_title="N(θ) [kN]",
    height=450
)

st.plotly_chart(fig_N, use_container_width=True)


# ============================================================
# Eccentricity plot
# ============================================================

st.header("Ексцентрицитет и проверка на ядрото")

fig_e = go.Figure()

fig_e.add_trace(go.Scatter(
    x=theta_deg,
    y=e,
    mode="lines+markers",
    name="e(θ) = M(θ) / N(θ)"
))

fig_e.add_trace(go.Scatter(
    x=theta_deg,
    y=np.full_like(theta, kernel_plus),
    mode="lines",
    name="+h/6"
))

fig_e.add_trace(go.Scatter(
    x=theta_deg,
    y=np.full_like(theta, kernel_minus),
    mode="lines",
    name="-h/6"
))

fig_e.add_hline(
    y=0,
    line_dash="dash"
)

fig_e.update_layout(
    title="Проверка: |e| ≤ h/6",
    xaxis_title="θ [градуси]",
    yaxis_title="e [m]",
    height=500
)

st.plotly_chart(fig_e, use_container_width=True)


# ============================================================
# Engineering conclusion
# ============================================================

st.header("Инженерска оценка")

number_critical = np.count_nonzero(~inside_kernel)
critical_ratio = number_critical / len(theta)
max_abs_e = np.nanmax(np.abs(e))

col_a, col_b, col_c = st.columns(3)

col_a.metric("Макс. |e| [m]", f"{max_abs_e:.4f}")
col_b.metric("h/6 [m]", f"{kernel_plus:.4f}")
col_c.metric("Критични сечения", f"{number_critical} / {len(theta)}")

if number_critical == 0:
    st.success(
        "Резултантата остава в ядрото на сечението за всички изчислени точки. "
        "Предварителна оценка: безопасно."
    )
elif critical_ratio < 0.25:
    st.warning(
        "Има отделни критични сечения, при които резултантата напуска ядрото. "
        "Предварителна оценка: необходим контрол."
    )
else:
    st.error(
        "Има значителен брой критични сечения, при които резултантата напуска ядрото. "
        "Предварителна оценка: необходимо укрепване."
    )

if np.any(critical_intrados):
    st.write("Препоръка: проверка/укрепване в зона към интрадоса.")

if np.any(critical_extrados):
    st.write("Препоръка: проверка/укрепване в зона към екстрадоса.")


# ============================================================
# Tables
# ============================================================

st.header("Критични сечения")

critical_df = df[df["Проверка"] != "В ядрото"]

if critical_df.empty:
    st.success("Няма критични сечения според проверката |e| ≤ h/6.")
else:
    st.dataframe(critical_df, use_container_width=True)

st.header("Пълна таблица с резултати")

st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Изтегли резултатите като CSV",
    data=csv,
    file_name="arch_results.csv",
    mime="text/csv"
)


# ============================================================
# Formula display
# ============================================================

st.header("Използвани формули")

#st.latex(r"R = \frac{L}{2}")
st.latex(r"L = 2R")
st.latex(r"V_A = V_B = \frac{q \pi R}{2}")
st.latex(r"H_A = H_B = \frac{qR}{\pi}")
st.latex(r"M_A = M_B = -qR^2 \left(1 - \frac{2}{\pi}\right)")

st.latex(
    r"M(\theta) = qR^2 \left("
    r"\frac{2}{\pi} - \cos\theta - \theta\sin\theta + "
    r"\frac{1}{\pi}\sin\theta"
    r"\right)"
)

st.latex(
    r"N(\theta) = -qR \left("
    r"\cos\theta + \theta\sin\theta + \frac{1}{\pi}\sin\theta"
    r"\right)"
)

st.latex(r"e(\theta) = \frac{M(\theta)}{N(\theta)}")
st.latex(r"|e| \leq \frac{h}{6}")