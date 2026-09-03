import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.constants import c, h, k, sigma


st.set_page_config(page_title="Blackbody Explorer", page_icon="☀️", layout="wide")


def planck_surface_flux(wavelength_angstrom, temperature):
    """Return pi*B_lambda in W m^-2 Angstrom^-1."""
    wavelength_m = np.asarray(wavelength_angstrom, dtype=float) * 1e-10
    exponent = h * c / (wavelength_m * k * temperature)
    # expm1 is accurate at small exponent; clipping prevents overflow warnings.
    denominator = np.expm1(np.clip(exponent, None, 700.0))
    b_lambda_per_m = (2.0 * h * c**2 / wavelength_m**5) / denominator
    return np.pi * b_lambda_per_m * 1e-10


def parse_temperatures(raw):
    values = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values or any(t <= 0 for t in values):
        raise ValueError("Enter one or more positive temperatures.")
    return values


def make_text_export(wavelength, curves, temperatures, areas):
    out = io.StringIO()
    out.write("# Blackbody Explorer export\n")
    out.write("# F_lambda = pi B_lambda [W m^-2 Angstrom^-1]\n")
    out.write("# temperatures_K = " + ", ".join(f"{t:g}" for t in temperatures) + "\n")
    for row in areas.itertuples(index=False):
        out.write(
            f"# T={row.Temperature_K:g} K: lambda_peak={row.Peak_Angstrom:.7g} A, "
            f"displayed_integral={row.Displayed_flux_W_m2:.7g} W m^-2, "
            f"bolometric_flux={row.Bolometric_flux_W_m2:.7g} W m^-2\n"
        )
    out.write("# wavelength_A " + " ".join(f"F_lambda_{t:g}K" for t in temperatures) + "\n")
    np.savetxt(out, np.column_stack([wavelength, *curves]), fmt="%.8e")
    return out.getvalue()


st.title("Blackbody Explorer")

with st.sidebar:
    st.header("Curves")
    temperature_text = st.text_input(
        "Temperatures (K)", "3000, 5800, 10000", help="Separate temperatures with commas."
    )
    st.header("Axes")
    x_min, x_max = st.slider(
        "Wavelength range (Å)", 1.0, 100000.0, (1000.0, 30000.0), step=100.0
    )
    log_x = st.checkbox("Logarithmic wavelength axis")
    log_y = st.checkbox("Logarithmic flux axis", value=True)
    auto_y = st.checkbox("Automatic flux range", value=True)
    if not auto_y:
        y_min = st.number_input("Minimum flux", min_value=0.0, value=0.0, format="%.3e")
        y_max = st.number_input("Maximum flux", min_value=0.0, value=1.0e4, format="%.3e")
    points = st.select_slider("Sampling", options=[500, 1000, 2000, 5000, 10000], value=2000)
    st.header("Annotations")
    show_peaks = st.checkbox("Show peaks", value=True)
    show_area = st.checkbox("Show area under each curve")

try:
    temperatures = parse_temperatures(temperature_text)
except ValueError as error:
    st.error(str(error))
    st.stop()

if x_min >= x_max:
    st.error("The maximum wavelength must be larger than the minimum wavelength.")
    st.stop()

if log_x:
    wavelength = np.geomspace(x_min, x_max, points)
else:
    wavelength = np.linspace(x_min, x_max, points)

fig = go.Figure()
curves = []
area_rows = []

for temperature in temperatures:
    flux = planck_surface_flux(wavelength, temperature)
    curves.append(flux)
    displayed_area = np.trapezoid(flux, wavelength)
    peak_angstrom = 2.897771955e7 / temperature
    area_rows.append(
        {
            "Temperature_K": temperature,
            "Peak_Angstrom": peak_angstrom,
            "Displayed_flux_W_m2": displayed_area,
            "Bolometric_flux_W_m2": sigma * temperature**4,
            "Displayed_fraction": displayed_area / (sigma * temperature**4),
        }
    )
    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=flux,
            mode="lines",
            name=f"{temperature:g} K",
            fill="tozeroy" if show_area else None,
            hovertemplate=(
                "λ = %{x:.6g} Å<br>Fλ = %{y:.6g} W m⁻² Å⁻¹"
                + f"<br>T = {temperature:g} K<extra></extra>"
            ),
        )
    )
    if show_peaks and x_min <= peak_angstrom <= x_max:
        peak_flux = planck_surface_flux(np.array([peak_angstrom]), temperature)[0]
        fig.add_trace(
            go.Scatter(
                x=[peak_angstrom], y=[peak_flux], mode="markers",
                marker=dict(size=10, symbol="diamond"),
                name=f"Peak: {temperature:g} K", showlegend=False,
                hovertemplate="Peak λ = %{x:.6g} Å<br>Fλ = %{y:.6g}<extra></extra>",
            )
        )

fig.update_layout(
    xaxis_title="Wavelength (Å)",
    yaxis_title="Surface spectral flux, Fλ (W m⁻² Å⁻¹)",
    xaxis_type="log" if log_x else "linear",
    yaxis_type="log" if log_y else "linear",
    hovermode="closest",
    template="plotly_white",
    height=650,
    margin=dict(l=70, r=30, t=30, b=70),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
if not auto_y:
    if y_max <= y_min or (log_y and y_min <= 0):
        st.error("Choose a valid Y range (and use a positive minimum for a log axis).")
        st.stop()
    fig.update_yaxes(range=[np.log10(y_min), np.log10(y_max)] if log_y else [y_min, y_max])

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "blackbody_spectra", "scale": 2}},
)
st.caption("Hover over a curve to see the wavelength and flux. Use the camera button in the plot toolbar to save a PNG.")

areas = pd.DataFrame(area_rows)
if show_area:
    shown = areas.rename(columns={
        "Temperature_K": "Temperature (K)",
        "Peak_Angstrom": "Peak wavelength (Å)",
        "Displayed_flux_W_m2": "Area in displayed range (W m⁻²)",
        "Bolometric_flux_W_m2": "Total area σT⁴ (W m⁻²)",
        "Displayed_fraction": "Fraction displayed",
    })
    st.dataframe(shown.style.format({col: "{:.6g}" for col in shown.columns}), hide_index=True, use_container_width=True)

text_export = make_text_export(wavelength, curves, temperatures, areas)
html_export = fig.to_html(full_html=True, include_plotlyjs=True)
download_1, download_2 = st.columns(2)
with download_1:
    st.download_button("Download figure (interactive HTML)", html_export, "blackbody_spectra.html", "text/html", use_container_width=True)
with download_2:
    st.download_button("Download data and results (text)", text_export, "blackbody_spectra.txt", "text/plain", use_container_width=True)

with st.expander("Physics and definitions"):
    st.markdown(
        r"""
The plotted quantity is the radiant flux leaving one square metre of a blackbody surface:

$$F_\lambda=\pi B_\lambda=\pi\frac{2hc^2}{\lambda^5}\frac{1}{e^{hc/(\lambda kT)}-1}. $$

Integrating over every wavelength gives $F=\sigma T^4$. The displayed-range area is computed numerically only between the selected wavelength limits. Wien's law gives the marked peak, $\lambda_{\rm max}T=2.8978\times10^7\ \mathrm{\AA\,K}$.
"""
    )
