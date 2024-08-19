# Probability_Distribution_Simulator

## Description

This Python application, built using Streamlit, offers a comprehensive tool for exploring and visualizing various probability distributions. Users can select a distribution type (continuous or discrete), input parameters, and examine statistical measures, probability mass functions/probability density functions, and simulated data.

## Installation

First, install the required libraries:

```bash
pip install numpy pandas matplotlib scipy streamlit
```
Save the code as a Python file (e.g., distribution_simulator.py).

## Usage
To run the application, use the following command:
```bash
streamlit run distribution_simulator.py
```

## Steps

1. Select a distribution type (Continuous or Discrete) from the sidebar.
2. Choose a specific distribution from the available options.
3. Input the required parameters for the chosen distribution.
4. View the statistical measures (mean, variance, skewness, kurtosis) in the main window.
5. Select the desired plots (PMF/PDF, Simulate Plot) from the sidebar.
6. Examine the generated plots in the main window.

## Features

- **Wide range of distributions:** Supports both continuous and discrete distributions.
- **Interactive interface:** Users can easily select distributions and input parameters.
- **Clear visualization:** Displays probability mass functions/probability density functions and simulated data.
- **Statistical calculations:** Provides mean, variance, skewness, and kurtosis for each distribution.
- **User-friendly:** Intuitive design and clear explanations.

## Imports

Necessary libraries are imported:
- **numpy** for numerical computations.
- **pandas** (potentially for data manipulation - not used here).
- **matplotlib.pyplot** for plotting.
- **scipy.stats** for statistical functions.
- **math** for mathematical operations.
- **itertools** for generating combinations (used in Multinomial).
- **streamlit** for building the web app.

## Distribution Classes

- Each class represents a specific probability distribution (e.g., Beta, Binomial).
- Each class constructor takes arguments for the distribution's parameters (e.g., `alpha` and `beta` for Beta).

### Methods within Each Class:

- `mean()`: Calculates the mean of the distribution.
- `variance()`: Calculates the variance of the distribution.
- `skewness()`: Calculates the skewness of the distribution (might be `None` if undefined).
- `kurtosis()`: Calculates the kurtosis of the distribution (might be `None` if undefined).
- `plot_pdf_pmf()`: Generates a plot of the distribution's Probability Density Function (PDF) for continuous distributions or Probability Mass Function (PMF) for discrete distributions.
- `simulate_data()`: Generates random samples from the distribution.

## Streamlit App

- The app title and description explain its functionality.
- The sidebar allows users to select the distribution type (Continuous or Discrete) and then choose a specific distribution from the selected category.
- A dictionary (`dist_classes`) maps distribution names to their corresponding classes for easy access.
- Based on the chosen distribution, the app prompts the user for specific parameters using `st.sidebar.number_input`.
- An instance of the corresponding distribution class is created with the provided parameters.
- For methods like `mean()`, `variance()`, etc., the app calculates the corresponding statistical measure.
- For `plot_pdf_pmf()` and `simulate_data()`, the app generates the respective plot or simulated data using Matplotlib and displays it.


## Example Usage

1. Select "Continuous" as the distribution type.
2. Choose "Beta" as the distribution.
3. Input `alpha=2` and `beta=5`.
4. View the calculated mean, variance, skewness, and kurtosis.
5. Select "PMF/PDF" and "Simulate Plot".
6. Observe the generated Beta distribution plots.



