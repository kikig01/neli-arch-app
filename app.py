import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# Page setup
# ============================================================

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
    value=4.00,
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

# ============================================================
# Structural formulas
# ============================================================

V_A = (q * np.pi * R) / 2
V_B = V_A

H_A = (q * R) / np.pi
H_B = H_A

M_A = -q * R**2 * (1 - 2 / np.pi)
M_B = M_A

# The formulas are valid from 0° to 90°.
# For the right half of the arch, mirror the angle:
# theta_calc = 0° at left support
# theta_calc = 90° at crown
# theta_calc = 0° at right support
theta_calc = np.minimum(theta, np.pi - theta)
theta_calc = np.minimum(theta, np.pi - theta)

M = q * R**2 * (
    2 / np.pi
    - np.cos(theta_calc)
    - theta_calc * np.sin(theta_calc)
    + (1 / np.pi) * np.sin(theta_calc)
)

N = -q * R * (
    np.cos(theta_calc)
    + theta_calc * np.sin(theta_calc)
    + (1 / np.pi) * np.sin(theta_calc)
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
# Helper functions
# ============================================================

def arc_xy(radius_values, theta_values):
    """
    Converts polar semicircle coordinates to Cartesian coordinates.
    theta = 0° left support, 90° crown, 180° right support.
    """
    x = -radius_values * np.cos(theta_values)
    y = radius_values * np.sin(theta_values)
    return x, y


def add_arch_reference_lines(fig, R_value, h_value, theta_values):
    """
    Adds intrados, extrados, and zero/axis line as background references.
    """
    r_axis = np.full_like(theta_values, R_value)
    r_intrados = np.full_like(theta_values, R_value - h_value / 2)
    r_extrados = np.full_like(theta_values, R_value + h_value / 2)

    x_axis_ref, y_axis_ref = arc_xy(r_axis, theta_values)
    x_intrados_ref, y_intrados_ref = arc_xy(r_intrados, theta_values)
    x_extrados_ref, y_extrados_ref = arc_xy(r_extrados, theta_values)

    fig.add_trace(go.Scatter(
        x=x_extrados_ref,
        y=y_extrados_ref,
        mode="lines",
        name="Екстрадос",
        line=dict(color="gray", width=2, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=x_intrados_ref,
        y=y_intrados_ref,
        mode="lines",
        name="Интрадос",
        line=dict(color="gray", width=2, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=x_axis_ref,
        y=y_axis_ref,
        mode="lines",
        name="Нулева линия / ос на арката",
        line=dict(color="red", width=2)
    ))


# ============================================================
# Pressure line for geometry plot
# ============================================================

# Convention:
# positive e goes towards intrados => smaller radius
# negative e goes towards extrados => larger radius

e_visual_geom = e.copy()
e_visual_geom[np.abs(e_visual_geom) > h / 2] = np.nan

r_pressure_visual = R - e_visual_geom
x_pressure_visual, y_pressure_visual = arc_xy(r_pressure_visual, theta)


# ============================================================
# Result data
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
        line=dict(width=3, color="black"),
        marker=dict(size=5),
        name="Линия на натиска"
    ))

# Span arrow: from one end of the intrados to the other end of the intrados
span_y = -0.16 * R

fig_geom.add_annotation(
    x=-R_intrados,
    y=span_y,
    ax=R_intrados,
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
    x=R_intrados,
    y=span_y,
    ax=-R_intrados,
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

# Rise arrow: from (0,0) to the intrados crown
fig_geom.add_annotation(
    x=0,
    y=R_intrados,
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
    ay=R_intrados,
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
    y=0.5 * R_intrados,
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

if show_pressure_line and np.any(np.abs(e) > h / 2):
    st.warning(
        "Част от реалната линия на натиска е извън дебелината на арката. "
        "В геометричната схема са показани само точките, които попадат в дебелината, "
        "за да не се деформира мащабът на чертежа."
    )


# ============================================================
# N(theta) polar semicircle diagram
# ============================================================

st.header("Полярна полудиаграма на нормалната сила N(θ)")

# In the formulas, N is mostly negative because of the sign convention.
# For the diagram, compression is shown as positive inward.
N_compression = -N

max_abs_N = np.nanmax(np.abs(N_compression))

if max_abs_N > 0:
    N_offset = (N_compression / max_abs_N) * (0.45 * h)
else:
    N_offset = np.zeros_like(N_compression)

# Positive compression goes towards intrados, therefore smaller radius
r_N = R - N_offset
x_N, y_N = arc_xy(r_N, theta)

fig_N_polar = go.Figure()
add_arch_reference_lines(fig_N_polar, R, h, theta)

fig_N_polar.add_trace(go.Scatter(
    x=x_N,
    y=y_N,
    mode="lines+markers",
    name="N(θ)",
    line=dict(color="blue", width=3),
    marker=dict(size=6),
    text=[
        f"θ = {td:.1f}°<br>N = {nv:.4f} kN"
        for td, nv in zip(theta_deg, N)
    ],
    hovertemplate="%{text}<extra></extra>"
))

fig_N_polar.update_layout(
    title="N(θ) като полярна полудиаграма",
    xaxis_title="x [m]",
    yaxis_title="y [m]",
    height=650,
    yaxis_scaleanchor="x",
    legend=dict(orientation="h", y=-0.18)
)

fig_N_polar.update_xaxes(
    range=[-R_extrados - 0.25 * R, R_extrados + 0.25 * R],
    zeroline=False
)

fig_N_polar.update_yaxes(
    range=[-0.20 * R, R_extrados + 0.35 * R],
    zeroline=False
)

st.plotly_chart(fig_N_polar, use_container_width=True)


# ============================================================
# M(theta) polar semicircle diagram
# ============================================================

st.header("Полярна полудиаграма на огъващия момент M(θ)")

max_abs_M = np.nanmax(np.abs(M))

if max_abs_M > 0:
    M_offset = (M / max_abs_M) * (0.45 * h)
else:
    M_offset = np.zeros_like(M)

# Positive M goes towards intrados, i.e. smaller radius.
# Negative M goes towards extrados, i.e. larger radius.
r_M = R - M_offset
x_M, y_M = arc_xy(r_M, theta)

fig_M_polar = go.Figure()
add_arch_reference_lines(fig_M_polar, R, h, theta)

fig_M_polar.add_trace(go.Scatter(
    x=x_M,
    y=y_M,
    mode="lines+markers",
    name="M(θ)",
    line=dict(color="purple", width=3),
    marker=dict(size=6),
    text=[
        f"θ = {td:.1f}°<br>M = {mv:.4f} kNm"
        for td, mv in zip(theta_deg, M)
    ],
    hovertemplate="%{text}<extra></extra>"
))

fig_M_polar.update_layout(
    title="M(θ): положителен момент към интрадоса",
    xaxis_title="x [m]",
    yaxis_title="y [m]",
    height=650,
    yaxis_scaleanchor="x",
    legend=dict(orientation="h", y=-0.18)
)

fig_M_polar.update_xaxes(
    range=[-R_extrados - 0.25 * R, R_extrados + 0.25 * R],
    zeroline=False
)

fig_M_polar.update_yaxes(
    range=[-0.20 * R, R_extrados + 0.35 * R],
    zeroline=False
)

st.plotly_chart(fig_M_polar, use_container_width=True)


# ============================================================
# Eccentricity polar semicircle diagram
# ============================================================

st.header("Полярна полудиаграма на ексцентрицитета e(θ)")

# Convention:
# positive e goes towards intrados => smaller radius
# negative e goes towards extrados => larger radius
#
# For visualisation only, hide points outside the arch thickness.
# The engineering check still uses the full e values.

e_visual = e.copy()
e_visual[np.abs(e_visual) > h / 2] = np.nan

r_e = R - e_visual

r_h6_plus = np.full_like(theta, R - h / 6)    # +h/6 towards intrados
r_h6_minus = np.full_like(theta, R + h / 6)   # -h/6 towards extrados

r_h3_plus = np.full_like(theta, R - h / 3)    # +h/3 towards intrados
r_h3_minus = np.full_like(theta, R + h / 3)   # -h/3 towards extrados

x_e, y_e = arc_xy(r_e, theta)

x_h6_plus, y_h6_plus = arc_xy(r_h6_plus, theta)
x_h6_minus, y_h6_minus = arc_xy(r_h6_minus, theta)

x_h3_plus, y_h3_plus = arc_xy(r_h3_plus, theta)
x_h3_minus, y_h3_minus = arc_xy(r_h3_minus, theta)

fig_e_polar = go.Figure()
add_arch_reference_lines(fig_e_polar, R, h, theta)

fig_e_polar.add_trace(go.Scatter(
    x=x_e,
    y=y_e,
    mode="lines+markers",
    name="e(θ)",
    line=dict(color="black", width=3),
    marker=dict(size=6),
    text=[
        f"θ = {td:.1f}°<br>e = {ev:.4f} m"
        for td, ev in zip(theta_deg, e)
    ],
    hovertemplate="%{text}<extra></extra>"
))

fig_e_polar.add_trace(go.Scatter(
    x=x_h6_plus,
    y=y_h6_plus,
    mode="lines",
    name="+h/6 към интрадоса",
    line=dict(color="green", width=2)
))

fig_e_polar.add_trace(go.Scatter(
    x=x_h6_minus,
    y=y_h6_minus,
    mode="lines",
    name="-h/6 към екстрадоса",
    line=dict(color="green", width=2, dash="dash")
))

fig_e_polar.add_trace(go.Scatter(
    x=x_h3_plus,
    y=y_h3_plus,
    mode="lines",
    name="+h/3 към интрадоса",
    line=dict(color="orange", width=2)
))

fig_e_polar.add_trace(go.Scatter(
    x=x_h3_minus,
    y=y_h3_minus,
    mode="lines",
    name="-h/3 към екстрадоса",
    line=dict(color="orange", width=2, dash="dash")
))

fig_e_polar.update_layout(
    title="e(θ): положителен ексцентрицитет към интрадоса",
    xaxis_title="x [m]",
    yaxis_title="y [m]",
    height=700,
    yaxis_scaleanchor="x",
    legend=dict(orientation="h", y=-0.22)
)

fig_e_polar.update_xaxes(
    range=[-R_extrados - 0.25 * R, R_extrados + 0.25 * R],
    zeroline=False
)

fig_e_polar.update_yaxes(
    range=[-0.20 * R, R_extrados + 0.35 * R],
    zeroline=False
)

st.plotly_chart(fig_e_polar, use_container_width=True)

if np.any(np.abs(e) > h / 2):
    st.warning(
        "Част от стойностите на e(θ) са извън дебелината на арката. "
        "В диаграмата са скрити точките извън сечението, за да остане чертежът четим."
    )


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
# Formula display
# ============================================================

st.header("Използвани формули")

st.latex(r"L = 2R")
st.latex(r"V_A = V_B = \frac{q \pi R}{2}")
st.latex(r"H_A = H_B = \frac{qR}{\pi}")
st.latex(r"M_A = M_B = -qR^2 \left(1 - \frac{2}{\pi}\right)")

st.latex(r"\theta^* = \min(\theta, \pi - \theta)")

st.latex(
    r"M(\theta) = qR^2 \left("
    r"\frac{2}{\pi} - \cos\theta^* - \theta^*\sin\theta^* + "
    r"\frac{1}{\pi}\sin\theta^*"
    r"\right)"
)

st.latex(
    r"N(\theta) = -qR \left("
    r"\cos\theta^* + \theta^*\sin\theta^* + \frac{1}{\pi}\sin\theta^*"
    r"\right)"
)

st.latex(r"e(\theta) = \frac{M(\theta)}{N(\theta)}")
st.latex(r"|e| \leq \frac{h}{6}")