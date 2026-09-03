# Blackbody Explorer

An interactive Streamlit app for plotting and comparing blackbody spectra in an astronomy class.

## Run it

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

For an existing research environment, consider creating a separate environment
for the app so its dependencies cannot change other astronomy packages.

The plot toolbar saves a PNG with its camera button. The buttons below the plot download an interactive HTML figure and a text file containing wavelengths, all plotted curves, peak locations, and displayed-range integrals.

Spectra are evaluated on a constant 1 Å wavelength grid. Selecting “Show area
under each curve” prints the numerical integral over the displayed wavelength
range below the plot.
