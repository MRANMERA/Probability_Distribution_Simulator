import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.stats import beta, binom, cauchy, gamma, geom, hypergeom, logistic, multinomial, nbinom, pareto, poisson, uniform, weibull_min
import math
import itertools
import streamlit as st
import cmath


class BetaDistribution:
    def __init__(self, a=2, b=5):
        self.a = a
        self.b = b

    def mean(self):
        return self.a / (self.a + self.b)

    def variance(self):
        return (self.a * self.b) / ((self.a + self.b) ** 2 * (self.a + self.b + 1))

    def skewness(self):
        return (2 * (self.b - self.a) * math.sqrt(self.a + self.b + 1)) / ((self.a + self.b + 2) * math.sqrt(self.a * self.b))

    def kurtosis(self):
        return (6 * ((self.a - self.b) ** 2 * (self.a + self.b + 1) - self.a * self.b * (self.a + self.b + 2))) / (self.a * self.b * (self.a + self.b + 2) * (self.a + self.b + 3))

    def plot_pdf_pmf(self):
        x = np.linspace(0, 1, 100)
        pdf = beta_dist.pdf(x, self.a, self.b)
        plt.plot(x, pdf, 'r-', lw=2)
        plt.title("Beta Distribution PDF")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

    def simulate_data(self, size=1000):
        data = np.random.beta(self.a, self.b, size)
        plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
        plt.title("Simulated Beta Data")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()
    
# Binomial Distribution Class
class BinomialDistribution:
    def __init__(self, n=10, p=0.5):
        self.n = n
        self.p = p

    def mean(self):
        return self.n * self.p

    def variance(self):
        return self.n * self.p * (1 - self.p)

    def skewness(self):
        return (1 - 2 * self.p) / math.sqrt(self.n * self.p * (1 - self.p))

    def kurtosis(self):
        return (1 - 6 * self.p * (1 - self.p)) / (self.n * self.p * (1 - self.p))

    def plot_pdf_pmf(self):
        x = np.arange(0, self.n + 1)
        pmf = binom.pmf(x, self.n, self.p)
        plt.bar(x, pmf)
        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.title('Binomial Distribution PMF')
        plt.show()

    def simulate_data(self, size=1000):
        simulated_data = np.random.binomial(self.n, self.p, size=size)
        plt.hist(simulated_data, bins=np.arange(0, self.n + 2)-0.5, density=True, alpha=0.75)
        plt.xlabel('x')
        plt.ylabel('Frequency')
        plt.title('Simulated Binomial Data')
        plt.show()

# Cauchy Distribution Class
class CauchyDistribution:
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def mean(self):
        return None  # Cauchy distribution does not have a mean

    def variance(self):
        return None  # Cauchy distribution does not have a variance

    def skewness(self):
        return None  # Undefined for Cauchy distribution

    def kurtosis(self):
        return None  # Undefined for Cauchy distribution


    def plot_pdf_pmf(self):
        x = np.linspace(self.loc - 4*self.scale, self.loc + 4*self.scale, 100)
        pdf = cauchy.pdf(x, self.loc, self.scale)
        plt.plot(x, pdf)
        plt.title("Cauchy Distribution PDF")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.show()

    def simulate_data(self, size=1000):
        data = np.random.standard_cauchy(size=size) * self.scale + self.loc
        plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
        plt.title("Simulated Cauchy Data")
        plt.show()

# Gamma Distribution Class
class GammaDistribution:
    def __init__(self, shape=2, scale=1):
        self.shape = shape
        self.scale = scale

    def mean(self):
        return self.shape * self.scale

    def variance(self):
        return self.shape * self.scale ** 2

    def skewness(self):
        return 2 / math.sqrt(self.shape)

    def kurtosis(self):
        return 6 / self.shape

    def plot_pdf_pmf(self):
        x = np.linspace(0, self.shape * self.scale * 3, 100)
        pdf = gamma.pdf(x, self.shape, scale=self.scale)
        plt.plot(x, pdf)
        plt.title("Gamma Distribution PDF")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.show()

    def simulate_data(self, size=1000):
        data = np.random.gamma(self.shape, self.scale, size)
        plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
        plt.title("Simulated Gamma Data")
        plt.show()

# Geometric Distribution Class
class GeometricDistribution:
    def __init__(self, p=0.5):
        self.p = p

    def mean(self):
        return 1 / self.p

    def variance(self):
        return (1 - self.p) / self.p ** 2

    def skewness(self):
        return (2 - self.p) / math.sqrt(1 - self.p)

    def kurtosis(self):
        return 6 + (self.p ** 2 / (1 - self.p))


    def plot_pdf_pmf(self):
        x = np.arange(1, 15)
        pmf = geom.pmf(x, self.p)
        plt.bar(x, pmf)
        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.title('Geometric Distribution PMF')
        plt.show()

    def simulate_data(self, size=1000):
        simulated_data = np.random.geometric(self.p, size=size)
        plt.hist(simulated_data, bins=np.arange(1, max(simulated_data) + 2)-0.5, density=True, alpha=0.75)
        plt.xlabel('x')
        plt.ylabel('Frequency')
        plt.title('Simulated Geometric Data')
        plt.show()

# Hypergeometric Distribution Class
class HypergeometricDistribution:
    def __init__(self, M=20, n=7, N=12):
        self.M = M  # Total population size
        self.n = n  # Total number of success states in the population
        self.N = N  # Number of draws

    def mean(self):
        return self.N * (self.n / self.M)

    def variance(self):
        return self.N * (self.n / self.M) * ((self.M - self.n) / self.M) * ((self.M - self.N) / (self.M - 1))

    def skewness(self):
        return (self.M - 2*self.n) * math.sqrt(self.M - 1) * (self.M - 2*self.N) / (math.sqrt(self.N * self.n * (self.M - self.n) * (self.M - self.N)) * (self.M - 2))

    def kurtosis(self):
        numerator = self.M**2 * (self.M + 1) - 4*self.M * (self.M + 1) * self.n + 6 * (self.n**2 * self.M + self.n) - 3 * self.n**2
        denominator = self.n * (self.M - self.n) * (self.M - 2) * (self.M - 3)
        return 1 + (numerator / denominator)


    def plot_pdf_pmf(self):
        x = np.arange(max(0, self.N + self.n - self.M), min(self.n, self.N) + 1)
        pmf = hypergeom.pmf(x, self.M, self.n, self.N)
        plt.bar(x, pmf)
        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.title('Hypergeometric Distribution PMF')
        plt.show()

    def simulate_data(self, size=1000):
        simulated_data = np.random.hypergeometric(self.n, self.M - self.n, self.N, size=size)
        plt.hist(simulated_data, bins=np.arange(min(simulated_data), max(simulated_data) + 2)-0.5, density=True, alpha=0.75)
        plt.xlabel('x')
        plt.ylabel('Frequency')
        plt.title('Simulated Hypergeometric Data')
        plt.show()

# Logistic Distribution Class
class LogisticDistribution:
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def mean(self):
        return self.loc

    def variance(self):
        return (self.scale ** 2) * (math.pi ** 2) / 3

    def skewness(self):
        return 0  # Logistic distribution is symmetric

    def kurtosis(self):
        return 6 / 5

    def plot_pdf_pmf(self):
        x = np.linspace(self.loc - 5 * self.scale, self.loc + 5 * self.scale, 100)
        pdf = logistic.pdf(x, loc=self.loc, scale=self.scale)
        plt.plot(x, pdf)
        plt.title("Logistic Distribution PDF")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.show()

    def simulate_data(self, size=1000):
        data = np.random.logistic(self.loc, self.scale, size)
        plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
        plt.title("Simulated Logistic Data")
        plt.show()

class MultinomialDistribution:
    def __init__(self, n=10, pvals=[0.2, 0.3, 0.5]):
        self.n = n
        self.pvals = pvals

    def mean(self):
        return [self.n * p for p in self.pvals]

    def variance(self):
        return [[self.n * p * (1 if i == j else 0 - q) for j, q in enumerate(self.pvals)] for i, p in enumerate(self.pvals)]

    def skewness(self):
        return None  # Skewness is more complex for Multinomial distribution

    def kurtosis(self):
        return None  # Kurtosis is more complex for Multinomial distribution

    def plot_pdf_pmf(self):
        # Generate all possible combinations of counts that sum to `self.n`
        categories = range(self.n + 1)
        all_combinations = [comb for comb in itertools.product(categories, repeat=len(self.pvals)) if sum(comb) == self.n]

        pmf_values = [multinomial.pmf(comb, self.n, self.pvals) for comb in all_combinations]
        
        # Convert the combinations to a more readable format for plotting
        x_labels = [str(comb) for comb in all_combinations]
        
        plt.bar(x_labels, pmf_values, color='blue')
        plt.xlabel('Categories')
        plt.ylabel('Probability')
        plt.title('Multinomial Distribution PMF')
        plt.xticks(rotation=90)
        plt.show()

    def simulate_data(self, size=1000):
        data = np.random.multinomial(self.n, self.pvals, size=size)
        
        # Plot each category in a different subplot
        fig, axs = plt.subplots(1, len(self.pvals), figsize=(15, 5), sharey=True)
        
        for i in range(len(self.pvals)):
            axs[i].hist(data[:, i], bins=np.arange(self.n + 2) - 0.5, density=True, alpha=0.6)
            axs[i].set_title(f"Category {i+1}")
            axs[i].set_xlabel("Count")
            axs[i].set_ylabel("Frequency")

        plt.suptitle("Simulated Multinomial Data")
        plt.tight_layout()
        plt.show()

# Negative Binomial Distribution Class
class NegativeBinomialDistribution:
    def __init__(self, n=5, p=0.5):
        self.n = n
        self.p = p

    def mean(self):
        return self.n * (1 - self.p) / self.p

    def variance(self):
        return self.n * (1 - self.p) / (self.p ** 2)

    def skewness(self):
        return (2 - self.p) / math.sqrt(self.n * (1 - self.p))

    def kurtosis(self):
        return 6 + self.p ** 2 / (self.n * (1 - self.p))

    def plot_pdf_pmf(self):
        x = np.arange(0, 20)
        pmf = nbinom.pmf(x, self.n, self.p)
        plt.bar(x, pmf)
        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.title('Negative Binomial Distribution PMF')
        plt.show()

    def simulate_data(self, size=1000):
        data = np.random.negative_binomial(self.n, self.p, size=size)
        plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
        plt.title("Simulated Negative Binomial Data")
        plt.show()

# Pareto Distribution Class
class ParetoDistribution:
    def __init__(self, b=2):
        self.b = b

    def mean(self):
        if self.b > 1:
            return self.b / (self.b - 1)
        return None  # Mean is undefined for b <= 1

    def variance(self):
        if self.b > 2:
            return (self.b / ((self.b - 1) ** 2 * (self.b - 2)))
        return None  # Variance is undefined for b <= 2

    def skewness(self):
        if self.b > 3:
            return (2 * (1 + self.b)) / (self.b - 3) * math.sqrt((self.b - 2) / self.b)
        return None  # Skewness is undefined for b <= 3

    def kurtosis(self):
        if self.b > 4:
            return 6 * (self.b ** 3 + self.b ** 2 - 6 * self.b - 2) / (self.b * (self.b - 3) * (self.b - 4))
        return None  # Kurtosis is undefined for b <= 4

    def plot_pdf_pmf(self):
        x = np.linspace(1, 10, 100)
        pdf = pareto.pdf(x, self.b)
        plt.plot(x, pdf)
        plt.title("Pareto Distribution PDF")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.show()

    def simulate_data(self, size=1000):
        data = np.random.pareto(self.b, size=size)
        plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
        plt.title("Simulated Pareto Data")
        plt.show()

# Poisson Distribution Class
class PoissonDistribution:
    def __init__(self, λ=3):
        self.λ = λ

    def mean(self):
        return self.λ

    def variance(self):
        return self.λ

    def skewness(self):
        return 1 / np.sqrt(self.λ)

    def kurtosis(self):
        return 1 / self.λ

    def plot_pdf_pmf(self):
        x = np.arange(0, 15)
        pmf = poisson.pmf(x, self.λ)
        plt.bar(x, pmf)
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Probability')
        plt.title('Poisson Distribution PMF')
        plt.show()

    def simulate_data(self, size=1000):
        simulated_data = np.random.poisson(self.λ, size=size)
        plt.hist(simulated_data, bins=np.arange(0, max(simulated_data) + 1)-0.5, density=True, alpha=0.75)
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Frequency')
        plt.title('Simulated Poisson Data')
        plt.show()

# Uniform Distribution Class
class UniformDistribution:
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    def mean(self):
        return (self.low + self.high) / 2

    def variance(self):
        return (self.high - self.low) ** 2 / 12

    def skewness(self):
        return 0  # Uniform distribution is symmetric

    def kurtosis(self):
        return -6 / 5

    def plot_pdf_pmf(self):
        x = np.linspace(self.low, self.high, 100)
        pdf = uniform.pdf(x, self.low, self.high - self.low)
        plt.plot(x, pdf)
        plt.title("Uniform Distribution PDF")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.show()

    def simulate_data(self, size=1000):
        data = np.random.uniform(self.low, self.high, size)
        plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
        plt.title("Simulated Uniform Data")
        plt.show()

# Weibull Distribution Class
class WeibullDistribution:
    def __init__(self, shape=2, scale=1):
        self.shape = shape
        self.scale = scale

    def mean(self):
        return self.scale * math.gamma(1 + 1/self.shape)

    def variance(self):
        return (self.scale ** 2) * (math.gamma(1 + 2/self.shape) - (math.gamma(1 + 1/self.shape) ** 2))

    def skewness(self):
        return (math.gamma(1 + 3/self.shape) - 3 * math.gamma(1 + 2/self.shape) * math.gamma(1 + 1/self.shape) + 2 * math.gamma(1 + 1/self.shape) ** 3) / (self.variance() ** (3/2))

    def kurtosis(self):
        return (math.gamma(1 + 4/self.shape) - 4 * math.gamma(1 + 3/self.shape) * math.gamma(1 + 1/self.shape) + 6 * math.gamma(1 + 2/self.shape) * math.gamma(1 + 1/self.shape) ** 2 - 3 * math.gamma(1 + 1/self.shape) ** 4) / (self.variance() ** 2) - 3


    def plot_pdf_pmf(self):
        x = np.linspace(0, self.scale * 5, 100)
        pdf = weibull_min.pdf(x, self.shape, scale=self.scale)
        plt.plot(x, pdf)
        plt.title("Weibull Distribution PDF")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.show()

    def simulate_data(self, size=1000):
        data = np.random.weibull(self.shape, size=size) * self.scale
        plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
        plt.title("Simulated Weibull Data")
        plt.show()

#streamlitapp
st.title(":yellow[_📈Probability Distribution Simulator💡_]")
st.write("When a user selects a distribution and provides its parameters, the app creates an instance of the corresponding class. This instance is then used to calculate the statistical measures and generate plots. When you choose a distribution (like Beta, Binomial, etc.) and give it specific numbers (like alpha=2, beta=5 for Beta), the app makes a copy of that distribution with those exact numbers. It then uses this copy to find things like the average, spread, and shape of the distribution. Finally, it shows you pictures (plots) that represent how this distribution looks.")

st.sidebar.title("Distribution Analyzer")
dist_type = st.sidebar.selectbox("Select Distribution Type", ["Continuous", "Discrete"])

# Dictionary to map distribution names to classes
dist_classes = {
    "Beta": BetaDistribution,
    "Binomial": BinomialDistribution,
    "Cauchy": CauchyDistribution,
    "Gamma": GammaDistribution,
    "Geometric": GeometricDistribution,
    "Hypergeometric": HypergeometricDistribution,
    "Logistic": LogisticDistribution,
    "Multinomial": MultinomialDistribution,
    "Negative Binomial": NegativeBinomialDistribution,
    "Pareto": ParetoDistribution,
    "Poisson": PoissonDistribution,
    "Uniform": UniformDistribution,
    "Weibull": WeibullDistribution,
}


if dist_type == "Continuous":
    dist_name = st.sidebar.selectbox("Select Continuous Distribution", [
        "Beta", "Cauchy", "Gamma", "Logistic", "Pareto", "Uniform", "Weibull"
    ])
elif dist_type == "Discrete":
    dist_name = st.sidebar.selectbox("Select Discrete Distribution", [
        "Binomial", "Geometric", "Hypergeometric", "Multinomial", "Negative Binomial", "Poisson"
    ])

# Handle the Beta distribution
if dist_name == "Beta":
    alpha = st.sidebar.number_input("Alpha:", value=1.0)
    beta = st.sidebar.number_input("Beta:", value=1.0)
    dist_instance = BetaDistribution(alpha, beta)

# Handle the Binomial distribution
elif dist_name == "Binomial":
    n = st.sidebar.number_input("Number of trials (n):", value=10)
    p = st.sidebar.number_input("Probability of success (p):", value=0.5)
    dist_instance = BinomialDistribution(n, p)

# Handle the Cauchy distribution
elif dist_name == "Cauchy":
    x0 = st.sidebar.number_input("Location (x0):", value=0.0)
    gamma = st.sidebar.number_input("Scale (gamma):", value=1.0)
    dist_instance = CauchyDistribution(x0, gamma)

# Handle the Gamma distribution
elif dist_name == "Gamma":
    k = st.sidebar.number_input("Shape parameter (k):", value=2.0)
    theta = st.sidebar.number_input("Scale parameter (theta):", value=2.0)
    dist_instance = GammaDistribution(k, theta)

# Handle the Geometric distribution
elif dist_name == "Geometric":
    p = st.sidebar.number_input("Probability of success (p):", value=0.5)
    dist_instance = GeometricDistribution(p)

# Handle the Hypergeometric distribution
elif dist_name == "Hypergeometric":
    N = st.sidebar.number_input("Total population size (N):", value=50)
    K = st.sidebar.number_input("Number of success states in population (K):", value=5)
    n = st.sidebar.number_input("Number of draws (n):", value=10)
    dist_instance = HypergeometricDistribution(N, K, n)

# Handle the Logistic distribution
elif dist_name == "Logistic":
    mu = st.sidebar.number_input("Location parameter (mu):", value=0.0)
    s = st.sidebar.number_input("Scale parameter (s):", value=1.0)
    dist_instance = LogisticDistribution(mu, s)

# Handle the Multinomial distribution
elif dist_name == "Multinomial":
    n = st.sidebar.number_input("Number of trials (n):", value=10)
    p = st.sidebar.text_input("Probabilities (comma-separated):", "0.2,0.3,0.5")
    p = [float(i) for i in p.split(',')]
    dist_instance = MultinomialDistribution(n, p)

# Handle the Negative Binomial distribution
elif dist_name == "Negative Binomial":
    r = st.sidebar.number_input("Number of successes (r):", value=5)
    p = st.sidebar.number_input("Probability of success (p):", value=0.5)
    dist_instance = NegativeBinomialDistribution(r, p)

# Handle the Pareto distribution
elif dist_name == "Pareto":
    alpha = st.sidebar.number_input("Shape parameter (alpha):", value=2.5)
    dist_instance = ParetoDistribution(alpha)

# Handle the Poisson distribution
elif dist_name == "Poisson":
    lam = st.sidebar.number_input("Rate (lambda):", value=3.0)
    dist_instance = PoissonDistribution(lam)

# Handle the Uniform distribution
elif dist_name == "Uniform":
    a = st.sidebar.number_input("Lower bound (a):", value=0.0)
    b = st.sidebar.number_input("Upper bound (b):", value=10.0)
    dist_instance = UniformDistribution(a, b)

# Handle the Weibull distribution
elif dist_name == "Weibull":
    shape = st.sidebar.number_input("Shape parameter:", value=1.5)
    scale = st.sidebar.number_input("Scale parameter:", value=2.0)
    dist_instance = WeibullDistribution(shape, scale)

# Display statistical measures and formulas
st.header(f"{dist_name} Distribution Statistical Measures")
st.write(f"The {dist_name} distribution is a {dist_type.lower()} distribution characterized by its parameters.")
st.write("### Values")
st.write("Mean:", dist_instance.mean())
st.write("Variance:", dist_instance.variance())
st.write("Skewness:", dist_instance.skewness())
st.write("Kurtosis:", dist_instance.kurtosis())

# Advanced plots
st.header(f"{dist_name} Distribution Plots")
plot_type = st.sidebar.multiselect("Select Plots to Display", ["PMF/PDF", "Simulate Plot"])

if "PMF/PDF" in plot_type:
    st.subheader("Probability Mass Function / Probability Density Function")
    fig, ax = plt.subplots()
    if dist_type == "Discrete":
        dist_instance.plot_pdf_pmf()
    else:
        dist_instance.plot_pdf_pmf()
    st.pyplot(fig)

if "Simulate Plot" in plot_type:
    st.subheader("Simulate Plot")
    fig, ax = plt.subplots()
    dist_instance.simulate_data(size=1000)
    st.pyplot(fig)

st.sidebar.markdown("This Project is created by Anmol Gupta [LinkedIn Profile](https://www.linkedin.com/in/anmol1701/)")