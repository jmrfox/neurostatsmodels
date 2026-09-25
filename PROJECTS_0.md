# Toy Projects for Statistical Neural Coding

A generative-model–first approach to sensory population codes.

## Table of Contents

- [Toy Projects for Statistical Neural Coding](#toy-projects-for-statistical-neural-coding)
  - [Table of Contents](#table-of-contents)
  - [Purpose](#purpose)
  - [Core Assumptions (Default Unless Stated Otherwise)](#core-assumptions-default-unless-stated-otherwise)
  - [Project 1: Gaussian Tuning Curves and Fisher Information](#project-1-gaussian-tuning-curves-and-fisher-information)
    - [1.1 Generative Model](#11-generative-model)
    - [1.2 Questions](#12-questions)
    - [1.3 Quantities to Compute](#13-quantities-to-compute)
    - [1.4 Key Lessons](#14-key-lessons)
    - [1.5 Failure Modes to Explore](#15-failure-modes-to-explore)
  - [Project 1.5: Resource Constraints and Efficient Coding](#project-15-resource-constraints-and-efficient-coding)
    - [1.5.1 Generative Model](#151-generative-model)
    - [1.5.2 Questions](#152-questions)
    - [1.5.3 Analyses](#153-analyses)
    - [1.5.4 Key Lessons](#154-key-lessons)
  - [Project 2: Spike Timing Codes via Coincidence Detection](#project-2-spike-timing-codes-via-coincidence-detection)
    - [2.1 Generative Model](#21-generative-model)
    - [2.2 Questions](#22-questions)
    - [2.3 Analyses](#23-analyses)
    - [2.4 Key Lessons](#24-key-lessons)
  - [Project 2.5: Temporal Integration Windows](#project-25-temporal-integration-windows)
    - [2.5.1 Generative Model](#251-generative-model)
    - [2.5.2 Questions](#252-questions)
    - [2.5.3 Analyses](#253-analyses)
    - [2.5.4 Key Lessons](#254-key-lessons)
  - [Project 3: Shared Variability via Latent Gain Modulation](#project-3-shared-variability-via-latent-gain-modulation)
    - [3.1 Generative Model](#31-generative-model)
    - [3.2 Questions](#32-questions)
    - [3.3 Analyses](#33-analyses)
    - [3.4 Key Lessons](#34-key-lessons)
  - [Project 3.5: Heterogeneous Noise](#project-35-heterogeneous-noise)
    - [3.5.1 Generative Model](#351-generative-model)
    - [3.5.2 Questions](#352-questions)
    - [3.5.3 Analyses](#353-analyses)
    - [3.5.4 Key Lessons](#354-key-lessons)
  - [Project 4: Population Geometry of Sound Location](#project-4-population-geometry-of-sound-location)
    - [4.1 Generative Model](#41-generative-model)
    - [4.2 Questions](#42-questions)
    - [4.3 Analyses](#43-analyses)
    - [4.4 Key Lessons](#44-key-lessons)
  - [Project 5: Model Mismatch and Sufficiency](#project-5-model-mismatch-and-sufficiency)
    - [5.1 Setup](#51-setup)
    - [5.2 Questions](#52-questions)
    - [5.3 Analyses](#53-analyses)
    - [5.4 Key Lessons](#54-key-lessons)
  - [Project 6: Learning and Adaptation in Population Codes](#project-6-learning-and-adaptation-in-population-codes)
    - [6.1 Generative Model](#61-generative-model)
    - [6.2 Questions](#62-questions)
    - [6.3 Analyses](#63-analyses)
    - [6.4 Key Lessons](#64-key-lessons)
  - [Project 7: Synthetic–Real Data Bridging](#project-7-syntheticreal-data-bridging)
    - [7.1 Strategy](#71-strategy)
    - [7.2 Examples](#72-examples)
    - [7.3 Key Lessons](#73-key-lessons)
  - [Project 8: Bayesian Decoding and Priors](#project-8-bayesian-decoding-and-priors)
    - [8.1 Generative Model](#81-generative-model)
    - [8.2 Questions](#82-questions)
    - [8.3 Analyses](#83-analyses)
    - [8.4 Key Lessons](#84-key-lessons)
  - [Project 9: Continuous vs Discrete Decoders](#project-9-continuous-vs-discrete-decoders)
    - [9.1 Setup](#91-setup)
    - [9.2 Questions](#92-questions)
    - [9.3 Analyses](#93-analyses)
    - [9.4 Key Lessons](#94-key-lessons)
  - [Toy Models to Develop](#toy-models-to-develop)
    - [Adaptive Gain Control](#adaptive-gain-control)
    - [Temporal Basis Functions](#temporal-basis-functions)
    - [Correlated Noise Generator](#correlated-noise-generator)
    - [Stimulus Trajectory Generator](#stimulus-trajectory-generator)
    - [Decoder Comparison Framework](#decoder-comparison-framework)
  - [Computational Best Practices](#computational-best-practices)
    - [Numerical Stability](#numerical-stability)
    - [Validation Strategies](#validation-strategies)
    - [Visualization Standards](#visualization-standards)
    - [Code Organization](#code-organization)
  - [Guiding Principles](#guiding-principles)
  - [Long-Term Goal](#long-term-goal)

## Purpose

This document outlines a sequence of toy modeling projects designed to build
fundamental intuition for **statistical approaches to neural coding**, with a
focus on sensory stimuli (especially auditory ITD/ILD).

The goal is not biological realism, but **conceptual clarity**:

- What information is represented?
- In what statistics?
- Under what assumptions?
- And how do those assumptions fail?

All projects emphasize **explicit generative models**:
stimulus → latent variables → spikes.

---

## Core Assumptions (Default Unless Stated Otherwise)

- Neurons are conditionally independent given latent variables
- Spikes follow Poisson or point-process statistics
- Tuning curves are smooth and parametric
- Decoding is local unless explicitly stated

Each project relaxes or breaks one of these assumptions.

---

## Project 1: Gaussian Tuning Curves and Fisher Information

### 1.1 Generative Model

- Stimulus: scalar $s$ (e.g. ITD)
- Population of neurons indexed by $i$
- Preferred stimuli $\mu_i$ tile stimulus space
- Firing rate:
$$r_i(s) = r_0 + r_{\max} \exp\left(-\frac{(s-\mu_i)^2}{2\sigma^2}\right)$$
- Spikes: Poisson with rate $r_i(s)$

---

### 1.2 Questions

- How does population Fisher information depend on tuning width $\sigma$?
- Why does FI diverge without constraints?
- What assumptions are required for an optimal $\sigma$?

---

### 1.3 Quantities to Compute

- Single-neuron FI:
$$J_i(s) = \frac{(\partial_s r_i(s))^2}{r_i(s)}$$
- Population FI:
$$J_{\text{pop}}(s) = \sum_i J_i(s)$$
- FI evaluated at fixed stimulus $s_0$
- Decoding variance vs. FI (MLE decoder)

---

### 1.4 Key Lessons

- Fisher information is **local** in stimulus space
- FI must be evaluated at fixed $s_0$
- Integrating FI over stimulus space has no decoding interpretation
- Optimal tuning widths only appear with:
  - population coverage
  - resource constraints

---

### 1.5 Failure Modes to Explore

- No firing-rate normalization → narrow tuning always wins
- Sparse populations → edge effects
- Numerical instability near zero firing rates

---

## Project 1.5: Resource Constraints and Efficient Coding

### 1.5.1 Generative Model

- Fixed total firing rate budget: $\sum_i r_i(s) = R_{\text{total}}$
- Or fixed metabolic cost: $\sum_i c_i r_i(s) = C$
- Optimize tuning parameters (e.g., $\sigma$, $\mu_i$ spacing) under constraint
- Spikes: Poisson with rate $r_i(s)$

---

### 1.5.2 Questions

- How does optimal $\sigma$ change with budget constraints?
- What is the trade-off between coverage and precision?
- Does efficient coding predict biological tuning widths?
- How should neurons be distributed across stimulus space?

---

### 1.5.3 Analyses

- Constrained optimization of FI
- Pareto frontiers (precision vs coverage)
- Comparison to biological data
- Sensitivity to constraint type

---

### 1.5.4 Key Lessons

- Constraints are necessary for optimal tuning to emerge
- Efficient coding provides normative predictions
- Trade-offs depend on stimulus statistics and task demands
- Metabolic costs can explain heterogeneous tuning

---

## Project 2: Spike Timing Codes via Coincidence Detection

### 2.1 Generative Model

- Stimulus: ITD + tone frequency
- Left/right inputs generate phase-locked spike trains
- Coincidence detector fires when inputs arrive within window $\Delta t$
- Spike jitter increases with noise

---

### 2.2 Questions

- When does timing outperform rate coding?
- How much jitter destroys ITD information?
- When does timing collapse into a rate code?

---

### 2.3 Analyses

- Mutual information (timing vs count)
- FI based on spike times
- GLMs with temporal filters
- Precision vs jitter

---

### 2.4 Key Lessons

- Timing codes are fragile under noise
- Rate codes often emerge as sufficient statistics
- Temporal precision has diminishing returns

---

## Project 2.5: Temporal Integration Windows

### 2.5.1 Generative Model

- Stimulus varies over time: $s(t)$ (e.g., dynamic ITD trajectory)
- Neurons integrate over window $\tau_i$: $\tilde{r}_i(t) = \int_{t-\tau_i}^{t} w(t') s(t') dt'$
- Firing rate: $r_i(t) = f(\tilde{r}_i(t))$ where $f$ is nonlinearity
- Spikes: Poisson with rate $r_i(t)$

---

### 2.5.2 Questions

- What integration window maximizes information about dynamic stimuli?
- When does temporal filtering improve vs degrade coding?
- How does optimal window size depend on stimulus statistics?
- Can heterogeneous windows improve population coding?

---

### 2.5.3 Analyses

- Mutual information for different $\tau$ values
- Fisher information for time-varying stimuli
- Optimal linear filters
- Comparison to biological temporal receptive fields

---

### 2.5.4 Key Lessons

- Optimal integration matches stimulus timescales
- Temporal filtering trades off temporal resolution for SNR
- Heterogeneous windows enable multi-scale coding
- Integration windows bridge rate and timing codes

---

## Project 3: Shared Variability via Latent Gain Modulation

### 3.1 Generative Model

$$r_i(s, t) = g_t \cdot f_i(s)$$

- $g_t$: trial-wise latent variable
- $f_i(s)$: tuning curve
- Spikes: Poisson conditioned on $g_t$

---

### 3.2 Questions

- How does shared gain affect decoding precision?
- Can the latent variable be inferred from spikes?
- When does variability help vs hurt coding?

---

### 3.3 Analyses

- Noise correlations
- Fisher information with latent variables
- Factor analysis / Poisson FA
- Demixed PCA

---

### 3.4 Key Lessons

- Correlations break FI additivity
- Variability can be structured and informative
- Latent-variable models explain overdispersion

---

## Project 3.5: Heterogeneous Noise

### 3.5.1 Generative Model

- Neurons have different noise levels: $\text{Var}(n_i | s) = \sigma_i^2 r_i(s)$ (Fano factor)
- Or signal-dependent noise: $\text{Var}(n_i | s) = \alpha_i + \beta_i r_i(s)$
- Correlations can be heterogeneous: $\text{Cov}(n_i, n_j | s) = \rho_{ij} \sqrt{\text{Var}(n_i)\text{Var}(n_j)}$
- Spikes: overdispersed Poisson or negative binomial

---

### 3.5.2 Questions

- How should decoder weight neurons with different reliability?
- Does heterogeneous noise change optimal tuning distributions?
- When do unreliable neurons still contribute information?
- How does noise heterogeneity interact with correlation structure?

---

### 3.5.3 Analyses

- Weighted Fisher information
- Optimal linear estimators (inverse-variance weighting)
- Information loss from ignoring noise structure
- Decoder robustness to noise model mismatch

---

### 3.5.4 Key Lessons

- Reliable neurons dominate information, but unreliable neurons still help
- Optimal decoders must account for noise heterogeneity
- Noise structure matters as much as mean responses
- Real populations have orders-of-magnitude variation in reliability

---

## Project 4: Population Geometry of Sound Location

### 4.1 Generative Model

- Stimulus: $(s_1, s_2) = (\text{ITD}, \text{ILD})$
- Neurons have 2D tuning surfaces
- Responses form manifolds in $\mathbb{R}^N$

---

### 4.2 Questions

- What is the dimensionality of the population code?
- How does noise reshape the manifold?
- How separable are different stimuli?

---

### 4.3 Analyses

- PCA / factor analysis
- Nonlinear embeddings (Isomap, UMAP)
- Representational similarity
- Curvature and distance metrics

---

### 4.4 Key Lessons

- Geometry matters more than individual tuning
- Low-dimensional structure emerges naturally
- Noise can flatten or distort manifolds

---

## Project 5: Model Mismatch and Sufficiency

### 5.1 Setup

- Generate data from a complex spiking model
- Fit simplified models:
  - Poisson GLM
  - LN model
  - rate-only decoder

---

### 5.2 Questions

- What structure survives model mismatch?
- Which statistics are invariant?
- When is a simple model sufficient?

---

### 5.3 Analyses

- Likelihood comparison
- Predictive performance
- Bias in parameter recovery
- Failure diagnostics

---

### 5.4 Key Lessons

- Models are lenses, not truths
- Sufficiency depends on task and noise
- Mis-specified models can still decode well

---

## Project 6: Learning and Adaptation in Population Codes

### 6.1 Generative Model

- Tuning curves adapt via Hebbian or error-driven rules
- Stimulus distribution changes over time
- Population reorganizes

---

### 6.2 Questions

- How do manifolds evolve during learning?
- Does FI increase, decrease, or redistribute?
- What statistics change first?

---

### 6.3 Analyses

- Time-varying FI
- Drift in latent dimensions
- Representational alignment
- Stability–plasticity tradeoffs

---

### 6.4 Key Lessons

- Learning reshapes geometry before tuning curves
- Population-level statistics change smoothly
- Adaptation is often low-dimensional

---

## Project 7: Synthetic–Real Data Bridging

### 7.1 Strategy

- Fit toy models to real auditory data
- Identify mismatches
- Extend models minimally

---

### 7.2 Examples

- Poisson → overdispersion → latent state
- Rate code → timing effects → temporal GLM
- Independent neurons → shared variability

---

### 7.3 Key Lessons

- Toys are hypotheses, not baselines
- Real data reveals which assumptions fail
- Minimal extensions are maximally informative

---

## Project 8: Bayesian Decoding and Priors

### 8.1 Generative Model

- Stimulus has prior distribution: $p(s)$ (e.g., uniform, Gaussian, or learned from data)
- Likelihood: $p(\mathbf{r}|s)$ from population response model
- Decoder combines: $p(s|\mathbf{r}) \propto p(\mathbf{r}|s) p(s)$
- Spikes: Poisson with rate $r_i(s)$

---

### 8.2 Questions

- How do priors reshape effective information?
- When does Bayesian decoding outperform maximum likelihood?
- What happens when the prior is mismatched?
- Can neural circuits implement Bayesian inference?

---

### 8.3 Analyses

- Posterior distributions for different priors
- Bias-variance trade-offs (Bayesian vs ML)
- Prior-dependent Fisher information
- Mutual information vs Fisher information
- Comparison to behavioral data

---

### 8.4 Key Lessons

- Priors can improve or degrade decoding depending on match
- Fisher information alone doesn't predict behavioral performance
- Bayesian framework explains perceptual biases
- Optimal coding depends on stimulus statistics

---

## Project 9: Continuous vs Discrete Decoders

### 9.1 Setup

- Same generative model, different decoder classes:
  - Linear decoders (weighted sum)
  - Population vector
  - Template matching (nearest neighbor)
  - Maximum likelihood / MAP
  - Neural network decoders
  - Kernel methods

---

### 9.2 Questions

- Which decoder class is sufficient for which generative model?
- How does decoder complexity trade off with data requirements?
- When do simple decoders fail catastrophically?
- Can we identify decoder class from neural/behavioral data?

---

### 9.3 Analyses

- Decoding error vs decoder complexity
- Sample complexity (data needed for each decoder)
- Robustness to model mismatch
- Computational cost vs performance
- Cross-validation and generalization

---

### 9.4 Key Lessons

- Simple decoders often perform surprisingly well
- Decoder choice matters most when data is limited
- Overfitting is a bigger problem than underfitting for neural data
- Biological plausibility constrains decoder class

---

## Toy Models to Develop

### Adaptive Gain Control

```python
class AdaptivePopulation(GaussianTunedPopulation):
    """Population with divisive normalization or gain adaptation"""
```

- Useful for Projects 3 and 6
- Models contrast adaptation, attention effects
- Implements: $r_i(s) = \frac{f_i(s)}{1 + \sum_j w_{ij} f_j(s)}$

### Temporal Basis Functions

```python
class TemporalBasisPopulation:
    """Neurons with different temporal filters (e.g., gamma functions)"""
```

- Useful for Projects 2 and 2.5
- Models temporal receptive fields
- Implements: $r_i(t) = \int h_i(\tau) s(t-\tau) d\tau$

### Correlated Noise Generator

```python
def generate_correlated_poisson(rates, correlation_matrix, n_trials):
    """Generate Poisson spikes with specified correlations"""
```

- Useful for Projects 3 and 3.5
- Essential for testing decoders under realistic noise
- Methods: copula-based, latent variable, or direct simulation

### Stimulus Trajectory Generator

```python
class StimulusTrajectory:
    """Generate realistic stimulus time series"""
```

- Useful for Projects 2.5 and 6
- Models: random walk, Ornstein-Uhlenbeck, naturalistic statistics
- Parameterized by timescales and smoothness

### Decoder Comparison Framework

```python
class DecoderBenchmark:
    """Compare multiple decoders on same data"""
```

- Useful for Projects 5 and 9
- Systematic comparison of decoder performance
- Includes cross-validation and error analysis

---

## Computational Best Practices

### Numerical Stability

- **Log-likelihoods**: Always work in log space for large populations
- **Underflow prevention**: Use `logsumexp` for normalization
- **Gradient checking**: Verify analytical gradients against numerical
- **Regularization**: Add small epsilon to avoid division by zero

### Validation Strategies

- **Synthetic data**: Test with known ground truth first
- **Cross-validation**: Use proper train/test splits
- **Posterior predictive checks**: Simulate from fitted model
- **Residual analysis**: Check for systematic patterns in errors

### Visualization Standards

- **Information landscapes**: FI as function of parameters (2D heatmaps)
- **Decoding error curves**: Error vs stimulus value or noise level
- **Correlation matrices**: Heatmaps with hierarchical clustering
- **Manifold visualizations**: PCA, t-SNE, UMAP projections
- **Tuning curve arrays**: Small multiples for population overview

### Code Organization

- **Modular design**: Separate generative models, decoders, and analyses
- **Reproducibility**: Set random seeds, save parameters
- **Documentation**: Docstrings with equations and references
- **Testing**: Unit tests for key functions
- **Version control**: Commit working code frequently

---

## Guiding Principles

- Always specify the generative model
- Distinguish local vs global metrics
- Impose explicit constraints
- Compare theory to decoding
- Treat failure modes as results

---

## Long-Term Goal

Develop intuition for:
- What neural statistics matter
- When information measures apply
- How population structure enables computation

This document is meant to evolve as experiments and insights accumulate.
