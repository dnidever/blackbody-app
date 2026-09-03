import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative
from scipy.constants import c, h, k, sigma


st.set_page_config(page_title="Blackbody Explorer", page_icon="☀️", layout="wide")


def planck_radiance(wavelength_angstrom, temperature):
    """Return B_lambda in W m^-2 sr^-1 Angstrom^-1."""
    wavelength_m = np.asarray(wavelength_angstrom, dtype=float) * 1e-10
    exponent = h * c / (wavelength_m * k * temperature)
    # expm1 is accurate at small exponent; clipping prevents overflow warnings.
    denominator = np.expm1(np.clip(exponent, None, 700.0))
    b_lambda_per_m = (2.0 * h * c**2 / wavelength_m**5) / denominator
    return b_lambda_per_m * 1e-10


def parse_temperatures(raw):
    values = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values or any(t <= 0 for t in values):
        raise ValueError("Enter one or more positive temperatures.")
    return values


DELTA_WAVELENGTH = 1.0  # Angstrom


def make_text_export(wavelength, curves, temperatures, results):
    out = io.StringIO()
    out.write("# Blackbody Explorer export\n")
    out.write("# B_lambda [W m^-2 sr^-1 Angstrom^-1]\n")
    out.write("# temperatures_K = " + ", ".join(f"{t:g}" for t in temperatures) + "\n")
    for row in results:
        out.write(
            f"# T={row['temperature']:g} K: lambda_peak={row['peak']:.7g} A, "
            f"total_flux={row['total_flux']:.7g} W m^-2\n"
        )
    out.write("# wavelength_A " + " ".join(f"B_lambda_{t:g}K" for t in temperatures) + "\n")
    np.savetxt(out, np.column_stack([wavelength, *curves]), fmt="%.8e")
    return out.getvalue()


st.title("Blackbody Explorer")

with st.sidebar:
    st.header("Curves")
    temperature_text = st.text_input(
        "Temperatures (K)", "4000, 5777, 7000", help="Separate temperatures with commas."
    )
    st.header("Axes")
    x_min, x_max = st.slider(
        "Wavelength range (Å)", 1.0, 100000.0, (1000.0, 30000.0), step=100.0
    )
    log_x = st.checkbox("Logarithmic wavelength axis")
    log_y = st.checkbox("Logarithmic radiance axis", value=False)
    set_y_range = st.checkbox("Set Y-axis range")
    if set_y_range:
        y_max = st.number_input("Maximum radiance", min_value=0.0, value=1.0e4, format="%.3e")
        y_min = st.number_input("Minimum radiance", min_value=0.0, value=0.0, format="%.3e")
    st.header("Annotations")
    show_peaks = st.checkbox("Show peaks", value=True)
    calculate_flux = st.checkbox("Calculate total flux (area under curve)")
    st.header("Fake filters")
    number_of_filters = st.number_input(
        "Number of top-hat filters", min_value=0, max_value=6, value=0, step=1
    )
    filters = []
    filter_defaults = [
        ("Blue", 3500.0, 4500.0),
        ("Visual", 5000.0, 6000.0),
        ("Red", 6500.0, 8000.0),
        ("Near-IR", 10000.0, 15000.0),
        ("Filter 5", 20000.0, 25000.0),
        ("Filter 6", 30000.0, 35000.0),
    ]
    for filter_number in range(int(number_of_filters)):
        default_name, default_min, default_max = filter_defaults[filter_number]
        filter_name = st.text_input(
            f"Filter {filter_number + 1} name", default_name, key=f"filter_name_{filter_number}"
        )
        filter_min, filter_max = st.slider(
            f"{filter_name} wavelength range (Å)",
            1.0,
            100000.0,
            (default_min, default_max),
            step=100.0,
            key=f"filter_range_{filter_number}",
        )
        filters.append((filter_name, filter_min, filter_max))

try:
    temperatures = parse_temperatures(temperature_text)
except ValueError as error:
    st.error(str(error))
    st.stop()

if x_min >= x_max:
    st.error("The maximum wavelength must be larger than the minimum wavelength.")
    st.stop()

wavelength = np.arange(x_min, x_max + 0.5 * DELTA_WAVELENGTH, DELTA_WAVELENGTH)

fig = go.Figure()
curves = []
results = []

for curve_number, temperature in enumerate(temperatures):
    color = qualitative.Plotly[curve_number % len(qualitative.Plotly)]
    radiance = planck_radiance(wavelength, temperature)
    curves.append(radiance)
    peak_angstrom = 2.897771955e7 / temperature
    results.append(
        {
            "temperature": temperature,
            "peak": peak_angstrom,
            "total_flux": sigma * temperature**4,
            "color": color,
        }
    )
    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=radiance,
            mode="lines",
            name=f"{temperature:g} K",
            line=dict(color=color),
            hovertemplate=(
                "λ = %{x:.6g} Å<br>Bλ = %{y:.6g} W m⁻² sr⁻¹ Å⁻¹"
                + f"<br>T = {temperature:g} K<extra></extra>"
            ),
        )
    )
    if show_peaks and x_min <= peak_angstrom <= x_max:
        fig.add_vline(
            x=peak_angstrom,
            line_dash="dash",
            line_color=color,
            line_width=1.5,
            opacity=0.75,
        )
        peak_radiance = planck_radiance(np.array([peak_angstrom]), temperature)[0]
        fig.add_trace(
            go.Scatter(
                x=[peak_angstrom], y=[peak_radiance], mode="markers",
                marker=dict(size=10, symbol="diamond", color=color),
                name=f"Peak: {temperature:g} K", showlegend=False,
                hovertemplate="Peak λ = %{x:.6g} Å<br>Bλ = %{y:.6g} W m⁻² sr⁻¹ Å⁻¹<extra></extra>",
            )
        )

filter_colors = qualitative.Set2
for filter_number, (filter_name, filter_min, filter_max) in enumerate(filters):
    fig.add_vrect(
        x0=filter_min,
        x1=filter_max,
        fillcolor=filter_colors[filter_number % len(filter_colors)],
        opacity=0.18,
        line_width=1.5,
        line_dash="dot",
        layer="below",
        annotation_text=filter_name,
        annotation_position="top left",
    )

fig.update_layout(
    xaxis_title="Wavelength (Å)",
    yaxis_title="Spectral radiance, Bλ (W m⁻² sr⁻¹ Å⁻¹)",
    xaxis_type="log" if log_x else "linear",
    yaxis_type="log" if log_y else "linear",
    hovermode="closest",
    template="plotly_white",
    height=650,
    margin=dict(l=70, r=30, t=30, b=70),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
if set_y_range:
    if y_max <= y_min or (log_y and y_min <= 0):
        st.error("Choose a valid Y range (and use a positive minimum for a log axis).")
        st.stop()
    fig.update_yaxes(range=[np.log10(y_min), np.log10(y_max)] if log_y else [y_min, y_max])

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "blackbody_spectra", "scale": 2}},
)
st.caption("Hover over a curve to see the wavelength and radiance. Use the camera button in the plot toolbar to save a PNG.")

if calculate_flux:
    st.markdown(r"**Total surface flux, $F=\pi\int_0^\infty B_\lambda\,d\lambda=\sigma T^4$**")
    for row in results:
        st.markdown(
            f"<span style='color:{row['color']}; font-size:1.15rem'>●</span> "
            f"<strong>{row['temperature']:g} K:</strong> {row['total_flux']:.6g} W m⁻²",
            unsafe_allow_html=True,
        )

text_export = make_text_export(wavelength, curves, temperatures, results)
html_export = fig.to_html(full_html=True, include_plotlyjs=True)
download_1, download_2 = st.columns(2)
with download_1:
    st.download_button("Download figure (interactive HTML)", html_export, "blackbody_spectra.html", "text/html", use_container_width=True)
with download_2:
    st.download_button("Download data and results (text)", text_export, "blackbody_spectra.txt", "text/plain", use_container_width=True)

with st.expander("Physics and definitions"):
    st.markdown(
        r"""
The plotted quantity is the blackbody spectral radiance:

$$B_\lambda=\frac{2hc^2}{\lambda^5}\frac{1}{e^{hc/(\lambda kT)}-1}. $$

Integrating $B_\lambda$ over every wavelength gives $\sigma T^4/\pi$. The displayed-range area is computed numerically only between the selected wavelength limits. Wien's law gives the marked peak, $\lambda_{\rm max}T=2.8978\times10^7\ \mathrm{\AA\,K}$.
"""
    )
