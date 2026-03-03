# Bayesian Estimation (Variational Inference)

This directory implements Bayesian parameter estimation using **Variational Inference (VI)** with NumPyro. The objective is to approximate posterior distributions in models where exact Bayesian inference is computationally intractable, particularly in high-dimensional transportation problems (e.g., OD matrices combined with behavioral parameters).

The framework is modular: the user provides a differentiable log-likelihood and log-prior, and the estimation engine performs stochastic variational inference using automatic differentiation and JAX acceleration.

---

## Model Assumptions

We assume a posterior of the form

\[
p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta),
\]

where:

- \( \theta \) is a vector of unconstrained parameters  
- \( y \) are observed measurements  

All parameters are expressed in an unconstrained space. Positivity constraints (e.g., OD flows) are enforced through smooth transformations such as exponentiation. This guarantees differentiability and stable gradient-based optimization.

Both the likelihood and prior must be differentiable with respect to the parameters to allow efficient automatic differentiation and JAX vectorization.

---

## Variational Inference and the ELBO

Exact Bayesian inference requires computing the marginal likelihood (the *evidence*)

\[
p(y) = \int p(y \mid \theta)\, p(\theta)\, d\theta,
\]

which is generally intractable in large models.

Instead, we approximate the posterior with a parametric distribution \( q_\phi(\theta) \). The parameters \( \phi \) are chosen to maximize the **Evidence Lower Bound (ELBO)**:

\[
\text{ELBO}(\phi)
=
\mathbb{E}_{q_\phi}[\log p(y,\theta)]
-
\mathbb{E}_{q_\phi}[\log q_\phi(\theta)].
\]

Maximizing the ELBO is equivalent to minimizing the Kullback–Leibler divergence

\[
\mathrm{KL}(q_\phi(\theta)\,\|\,p(\theta \mid y)).
\]

Optimization is performed using stochastic gradients and the Adam optimizer through NumPyro’s SVI interface.

The ELBO trace reported during estimation serves as a convergence diagnostic.

---

## Prior Structure

The framework supports flexible prior specifications. In OD estimation, positivity is typically enforced via a log-deviation parameterization:

\[
f_k = f_{0,k} \exp(z_k),
\]

with

\[
z_k \sim \mathcal{N}(0, \sigma_z^2).
\]

This induces a lognormal prior centered on a baseline OD matrix \( f_0 \). The scale parameter \( \sigma_z \) controls the degree of regularization:

- Small \( \sigma_z \): strong shrinkage toward \( f_0 \)  
- Large \( \sigma_z \): weakly informative prior  

The prior therefore acts as a calibration prior rather than a structural demand model.

Two conventions are supported:

- Direct specification of the full log prior  
- Specification relative to a standard normal base distribution (for compatibility with NumPyro autoguides)

---

## Variational Families

The posterior approximation is Gaussian in the unconstrained parameter space. Supported guide families include:

- Diagonal Gaussian (scalable, robust)
- Low-rank Gaussian (captures partial correlations)
- Full multivariate Gaussian (computationally expensive)

For high-dimensional OD problems, diagonal or low-rank approximations are recommended.

---

## Why Variational Inference?

Variational inference is chosen for scalability. Transportation calibration problems can involve hundreds or thousands of parameters. Markov Chain Monte Carlo (MCMC) methods, while more accurate asymptotically, often become computationally prohibitive in such settings.

VI provides:

- Fast approximate posterior inference  
- Posterior means and variances  
- Posterior samples  
- Practical uncertainty quantification  

The method prioritizes computational efficiency and robustness over exact asymptotic accuracy.

---

## Limitations

Variational inference minimizes \( \mathrm{KL}(q \| p) \), which typically underestimates posterior variance. It may struggle with strongly multimodal posteriors, and convergence can depend on initialization and guide choice.

ELBO stabilization and sensitivity analysis should therefore be used routinely.

---

## Alternative Approaches

Other estimation strategies may be considered depending on the problem:

**Markov Chain Monte Carlo (e.g., NUTS)**  
Provides more accurate posterior exploration but at significantly higher computational cost.

**Laplace Approximation**  
A second-order Gaussian approximation around a mode. Fast but purely local.

**Maximum a Posteriori (MAP)**  
Deterministic regularized estimation without uncertainty quantification.

---

## Intended Use

This module is designed for large-scale calibration problems in transportation modeling, including joint estimation of OD demand and behavioral parameters. It emphasizes modularity, scalability, and seamless integration with JAX-based assignment and likelihood computations.