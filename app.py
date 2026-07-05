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
# Helper function for semicircle polar-style diagrams
# ============================================================

def semicircle_trace(theta, r, name, hover_values=None):
    """
    Draws a polar-style semicircle in Cartesian coordinates.
    theta is in radians, r is the radial value.
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    if hover_values is None:
        hover_values = r

    return go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=name,
        text=[
            f"θ = {td:.1f}°<br>value = {val:.4f}"
            for td, val in zip(theta_deg, hover_values)
        ],
        hovertemplate="%{text}<extra></extra>"
    )

# ============================================================
# Helper functions for engineering-style polar semicircle diagrams
# ============================================================

def arc_xy(radius_values, theta):
    """
    Converts polar semicircle coordinates to Cartesian coordinates.
    theta = 0° left support, 90° crown, 180° right support.
    """
    x = -radius_values * np.cos(theta)
    y = radius_values * np.sin(theta)
    return x, y


def add_arch_reference_lines(fig, R, h, theta):
    """
    Adds intrados, extrados, and zero/axis line as background references.
    """
    r_axis = np.full_like(theta, R)
    r_intrados = np.full_like(theta, R - h / 2)
    r_extrados = np.full_like(theta, R + h / 2)

    x_axis_ref, y_axis_ref = arc_xy(r_axis, theta)
    x_intrados_ref, y_intrados_ref = arc_xy(r_intrados, theta)
    x_extrados_ref, y_extrados_ref = arc_xy(r_extrados, theta)

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
# Helper for separate scaled polar diagrams
# ============================================================

def scaled_polar_semicircle(theta, values, theta_deg, name, unit, line_color):
    """
    Creates a separate engineering-style semicircular polar diagram.
    The radius is scaled from the absolute value.
    The real signed value is kept in hover text.
    """

    values_abs = np.abs(values)
    max_value = np.nanmax(values_abs)

    if max_value > 0:
        r = values_abs / max_value
    else:
        r = np.zeros_like(values_abs)

    # Semicircle coordinates:
    # theta = 0° left, theta = 90° top, theta = 180° right
    x = -r * np.cos(theta)
    y = r * np.sin(theta)

    fig = go.Figure()

    # Reference semicircles: 25%, 50%, 75%, 100%
    for rr, label in zip([0.25, 0.50, 0.75, 1.00], ["25%", "50%", "75%", "100%"]):
        x_ref = -rr * np.cos(theta)
        y_ref = rr * np.sin(theta)

        fig.add_trace(go.Scatter(
            x=x_ref,
            y=y_ref,
            mode="lines",
            name=label,
            line=dict(color="lightgray", width=1, dash="dot"),
            hoverinfo="skip"
        ))

    # Actual scaled diagram
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=name,
        line=dict(color=line_color, width=3),
        marker=dict(size=6),
        text=[
            f"θ = {td:.1f}°<br>{name} = {v:.4f} {unit}<br>|{name}| / max = {rv:.3f}"
            for td, v, rv in zip(theta_deg, values, r)
        ],
        hovertemplate="%{text}<extra></extra>"
    ))

    fig.update_layout(
        title=f"{name} като отделна мащабирана полярна полудиаграма",
        xaxis_title="мащабиран x",
        yaxis_title="мащабиран y",
        height=600,
        yaxis_scaleanchor="x",
        legend=dict(orientation="h", y=-0.15)
    )

    fig.update_xaxes(
        range=[-1.15, 1.15],
        zeroline=True
    )

    fig.update_yaxes(
        range=[-0.10, 1.15],
        zeroline=True
    )

    return fig

# ============================================================
# N(theta) separate scaled polar diagram
# ============================================================

st.header("Полярна полудиаграма на нормалната сила N(θ)")

fig_N_scaled = scaled_polar_semicircle(
    theta=theta,
    values=N,
    theta_deg=theta_deg,
    name="N(θ)",
    unit="kN",
    line_color="blue"
)

st.plotly_chart(fig_N_scaled, use_container_width=True)

# ============================================================
# M(theta) separate scaled polar diagram
# ============================================================

st.header("Полярна полудиаграма на огъващия момент M(θ)")

fig_M_scaled = scaled_polar_semicircle(
    theta=theta,
    values=M,
    theta_deg=theta_deg,
    name="M(θ)",
    unit="kNm",
    line_color="purple"
)

st.plotly_chart(fig_M_scaled, use_container_width=True)

# ============================================================
# Eccentricity polar semicircle diagram
# ============================================================

st.header("Полярна полудиаграма на ексцентрицитета e(θ)")

# Positive e goes towards intrados => smaller radius.
# Negative e goes towards extrados => larger radius.

# For visualisation only:
# hide points that are far outside the section, so the plot stays readable.
e_visual = e.copy()
e_visual[np.abs(e_visual) > h / 2] = np.nan

r_e = R - e_visual

r_h6_plus = np.full_like(theta, R - h / 6)   # +h/6 towards intrados
r_h6_minus = np.full_like(theta, R + h / 6)  # -h/6 towards extrados

r_h3_plus = np.full_like(theta, R - h / 3)   # +h/3 towards intrados
r_h3_minus = np.full_like(theta, R + h / 3)  # -h/3 towards extrados

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