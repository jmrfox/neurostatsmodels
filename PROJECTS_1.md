# Spiking Neural Network Project Guide

## Purpose

This project sequence is intended to build practical intuition for **recurrent spiking neural networks as dynamical and computational systems**.

The emphasis is not on basic single-neuron modeling or introductory artificial neural networks. Assume familiarity with:

- integrate-and-fire and related neuron models
- numerical simulation of single neurons
- basic parameter fitting
- conventional artificial neural networks
- standard Python scientific computing

The central questions throughout the sequence are:

1. **Construct:** How can recurrent spiking networks implement useful computations?
2. **Analyze:** How can we characterize the dynamics of those networks?
3. **Infer:** Given observations of population activity, what can we recover about the underlying system?

Projects should start small enough that the dynamics can be understood almost completely, then progress toward larger networks, learned recurrent dynamics, system identification, and comparison with real neural data.

---

# General Notebook Philosophy

Each project should be implemented as a self-contained **Marimo notebook** if practical. Jupyter is an acceptable fallback when a library does not interact cleanly with Marimo.

The notebooks should be pedagogical rather than production software.

Do **not** impose a particular software architecture. In particular, do not prescribe specific classes, methods, modules, inheritance structures, or APIs unless they arise naturally from the problem.

Favor clarity, experimentation, and mathematical transparency over abstraction.

## Interactive exploration

Use Marimo's reactive and interactive features when they genuinely help build intuition.

Useful controls may include:

- sliders for coupling strength
- sliders for noise level
- network size
- E/I ratio
- synaptic time constants
- recurrent gain
- external input strength
- simulation duration
- sparsity
- reservoir spectral scale or analogous recurrence parameters
- learning rates
- perturbation magnitude

Interactive controls should expose scientifically interesting parameters rather than merely demonstrate UI functionality.

Where possible, changing a parameter should immediately update:

- raster plots
- population firing rates
- state-space trajectories
- tuning curves
- decoding performance
- attractor structure
- dimensionality measures
- other relevant diagnostics

Avoid making notebooks excessively complicated merely to make them interactive.

---

# Common Analysis Toolbox

The projects should progressively reuse a common conceptual toolbox.

Relevant analyses include:

- spike raster plots
- firing-rate estimation
- PSTHs
- ISI distributions
- Fano factors
- autocorrelation
- cross-correlation
- population firing rates
- population vectors
- covariance and correlation matrices
- PCA
- dimensionality estimates
- state-space trajectories
- phase portraits
- fixed-point reasoning
- perturbation analysis
- decoding
- classification
- regression
- likelihood-based model comparison
- parameter recovery
- connectivity recovery
- trial-to-trial variability

Not every project needs every analysis.

The important goal is to increasingly treat the spike trains as observations of an underlying **dynamical system**, rather than simply as collections of firing rates.

---

# Suggested Libraries

Use libraries based on the needs of each project rather than forcing one framework everywhere.

Useful options include:

- **Brian2** for explicit mechanistic spiking-network simulations
- **snnTorch** for differentiable and surrogate-gradient-trained SNNs
- **PyTorch** when custom differentiable dynamics are helpful
- NumPy / SciPy for analysis and smaller transparent implementations
- scikit-learn for decoding, dimensionality reduction, and simple statistical models
- matplotlib / Plotly for visualization

Whenever possible, keep the mathematical relationship between the equations and the implementation visible.

A library should not become a black box.

---

# Project 10 — Recurrent Excitation, Inhibition, and Network Regimes

## Goal

Develop intuition for how simple recurrent connectivity produces qualitatively different population dynamics.

## Network

Construct a modest recurrent spiking network containing excitatory and inhibitory neurons.

Start with simple random connectivity.

## Explore

Systematically vary:

- recurrent excitation
- recurrent inhibition
- external drive
- network sparsity
- synaptic time constants
- noise

Observe transitions between regimes such as:

- quiescent activity
- stable asynchronous firing
- highly synchronized firing
- oscillatory activity
- runaway excitation
- inhibition-dominated suppression

## Analysis

For each regime examine:

- raster plots
- population firing rates
- E and I population rates separately
- ISI distributions
- firing-rate distributions
- synchrony measures
- autocorrelation
- population-rate power spectra

## Key question

How much can be inferred about network state from the raster plot alone, and what additional information appears when the activity is treated as a population dynamical signal?

## Extension

Construct a two-dimensional "phase diagram" over excitatory and inhibitory coupling strength and classify the observed network regimes.

---

# Project 11 — Persistent Activity and Bistability

## Goal

Build the simplest recurrent spiking circuit that behaves as a memory element.

## Network

Construct a recurrent population with sufficiently strong positive feedback that it can support two distinguishable states, for example:

- low activity
- persistent high activity

Use a transient external stimulus to switch between states.

## Experiments

Test:

- stimulus duration
- stimulus strength
- recurrent coupling
- noise
- inhibitory stabilization

Determine when persistent activity survives after the external input disappears.

## Analysis

Visualize:

- raster plots
- population rate
- trials from different initial conditions
- distributions of final network states
- state transitions

Treat population firing rate as an approximate low-dimensional state variable.

## Key questions

- When does recurrent excitation create memory?
- What destroys the memory?
- How sharply separated are the states?
- Does the spiking network behave approximately like a bistable dynamical system?

## Extension

Apply brief perturbations during the persistent state and measure whether the network returns to the same state or switches to the other attractor.

---

# Project 12 — Winner-Take-All and Competition

## Goal

Study computation through competition between recurrent neural populations.

## Network

Construct two or more excitatory populations coupled through shared or cross-population inhibition.

Each population should represent a competing state or choice.

## Experiments

Provide:

- equal inputs
- weakly biased inputs
- noisy inputs
- time-varying inputs

Explore how recurrent excitation and inhibition influence competition.

## Analysis

Measure:

- decision time
- winning population
- reliability
- sensitivity to input difference
- sensitivity to noise
- hysteresis

Plot trajectories using population firing rates as state variables.

## Key question

How does a network convert a small input difference into a categorical population state?

## Extension

Interpret the circuit as:

- a decision circuit
- a maximum-selection operation
- a primitive computational element

---

# Project 13 — Discrete Attractor Memory

## Goal

Move from bistability to associative memory.

## Network

Construct a recurrent spiking network containing several stored activity patterns.

Patterns may initially be simple structured subsets of neurons rather than a biologically detailed memory model.

## Experiments

Present:

- complete stored patterns
- corrupted patterns
- partial cues
- mixtures of patterns
- random inputs

Determine whether network dynamics converge toward stored states.

## Analysis

Define an overlap measure between current population activity and each stored pattern.

Plot overlap as a function of time.

Explore:

- basin size
- pattern interference
- storage capacity
- noise robustness
- connectivity strength

## Key questions

- What does an attractor look like in spike-based population activity?
- How does cue completion appear dynamically?
- How does increasing the number of memories alter stability?

---

# Project 14 — Continuous Attractors and Ring Networks

## Goal

Study networks whose stable states form a continuum rather than a discrete set.

## Network

Construct a ring-attractor-like spiking network in which neurons have preferred positions or angles and connectivity depends on distance around the ring.

The network should support a localized activity bump.

## Experiments

Test:

- initialization at different positions
- transient inputs
- noisy input
- asymmetric connectivity
- changes in recurrent width
- changes in inhibition

## Analysis

Decode the represented angle from population activity.

Track:

- bump position
- bump width
- bump amplitude
- drift
- diffusion under noise

## Key questions

- How can a recurrent network represent a continuous variable?
- Why is the family of bump states an attractor manifold?
- How does noise move activity along versus away from the manifold?

## Extension

Introduce a velocity-like input that moves the bump around the ring, producing a simple model of neural integration.

---

# Project 15 — Evidence Accumulation

## Goal

Construct a spiking circuit that performs temporal integration.

## Task

Provide noisy streams of evidence favoring one of two alternatives.

The network must integrate information over time and produce a decision.

## Experiments

Vary:

- evidence strength
- evidence duration
- noise
- recurrent coupling
- integration time constant

## Analysis

Measure:

- accuracy
- decision time
- psychometric curves
- chronometric curves
- population trajectories

Compare individual trials with trial-averaged behavior.

## Key questions

- How does integration emerge from recurrent spiking dynamics?
- What determines the effective integration time scale?
- When does the circuit behave like a drift-diffusion process?

## Extension

Attempt to extract an approximate one-dimensional decision variable from the full population activity.

---

# Project 16 — Spiking Reservoir Computing

## Goal

Use recurrent spiking dynamics as a general nonlinear temporal representation.

## Network

Construct a randomly connected recurrent spiking reservoir.

Keep recurrent weights fixed.

Train only a simple readout.

## Tasks

Start with several temporal tasks of increasing difficulty, such as:

- reconstructing a recent input
- delayed signal reconstruction
- nonlinear transformation of an input signal
- temporal pattern classification
- combining information arriving at different times

## Analysis

Measure performance while varying:

- reservoir size
- sparsity
- recurrent strength
- synaptic time constants
- E/I balance
- noise

## Key concepts

Explore:

- fading memory
- temporal separation
- nonlinear expansion
- sensitivity to perturbations
- richness of reservoir states

## Important analysis

Measure **memory capacity** systematically.

Determine how long past input remains decodable from the current network state.

## Extension

Compare reservoir performance with dimensionality of population activity.

Ask whether better computational performance corresponds to richer or higher-dimensional internal dynamics.

---

# Project 17 — Reservoir Dynamical Regimes

## Goal

Study why some recurrent networks make better reservoirs than others.

This project should focus more heavily on analysis than task performance.

## Experiments

Sweep recurrence parameters from weak to strong.

Identify regimes such as:

- strongly contracting dynamics
- useful fading-memory dynamics
- oscillatory dynamics
- unstable or chaotic-like dynamics

## Analysis

Use repeated trials with slightly perturbed initial conditions or inputs.

Measure how quickly trajectories:

- converge
- remain separated
- diverge

Examine:

- population dimensionality
- autocorrelation time
- decoding memory
- task performance
- sensitivity to perturbation

## Key question

Is useful reservoir computation associated with an intermediate dynamical regime between excessive stability and excessive instability?

Do not assume the answer in advance; test it.

---

# Project 18 — Trained Recurrent Spiking Network

## Goal

Train recurrent spiking dynamics to solve a temporal task.

Use surrogate gradients or another practical SNN training method.

## Task

Begin with a simple delayed-response or working-memory problem.

For example:

1. present a brief cue
2. remove the cue
3. wait through a delay
4. require the network to report the cue afterward

## Analysis

Do not stop at training accuracy.

Examine what the network learned.

Analyze:

- population firing rates
- raster plots
- PCA trajectories
- delay-period dynamics
- trial-to-trial variability
- trajectories conditioned on cue identity

## Key questions

- Is information stored through persistent activity?
- Is it stored through dynamic trajectories?
- Is it stored in a low-dimensional subspace?
- Does the network develop attractor-like states?

## Extension

Train several independently initialized networks on the same task and compare the dynamical strategies they discover.

---

# Project 19 — Context-Dependent Computation

## Goal

Study recurrent SNNs whose computation depends on internal or external context.

## Task

Construct a task in which identical sensory input requires different output depending on a context cue.

Examples:

- integrate one of two input channels
- respond to one feature while ignoring another
- switch between two mappings

## Analysis

Study how context changes population trajectories.

Look for:

- context-dependent subspaces
- gating
- trajectory divergence
- shared versus context-specific dimensions

Try decoding:

- stimulus
- context
- output

from different stages of the population activity.

## Key question

How does a recurrent network dynamically route the same input into different computations?

---

# Project 20 — Sequence Generation

## Goal

Study autonomous recurrent dynamics that generate structured temporal activity.

## Task

Train or construct a recurrent SNN that produces a reproducible sequence after a brief trigger.

Possible outputs include:

- sequential activation of neural groups
- a continuous temporal signal
- a repeating pattern
- several sequences selected by different cues

## Analysis

Examine:

- trial-to-trial timing variability
- robustness to perturbation
- speed variation
- trajectory geometry
- dimensionality
- temporal decoding

## Key questions

- Is the sequence generated by a chain-like mechanism?
- Is it better understood as a trajectory through a continuous dynamical system?
- What happens if activity is perturbed midway through the sequence?

---

# Project 21 — Population Geometry and Neural Manifolds

## Goal

Develop intuition for geometric analysis of spiking population activity.

Use networks from previous projects rather than necessarily constructing a new network.

## Analysis

Convert population activity into suitable time-dependent feature representations and examine:

- PCA
- participation ratio
- dimensionality versus time
- trajectories in latent space
- condition-dependent trajectories
- perturbation recovery
- attractor geometry

Compare networks implementing:

- bistability
- continuous attractors
- evidence accumulation
- working memory
- reservoir computation

## Key questions

- Which computational variables correspond to low-dimensional directions?
- When does high-dimensional spiking activity collapse onto low-dimensional dynamics?
- How do discrete and continuous attractors appear geometrically?
- How does task complexity affect dimensionality?

## Important caution

Do not equate a visually attractive PCA plot with evidence of a true manifold.

Use quantitative diagnostics where possible.

---

# Project 22 — Perturbation-Based Dynamical Analysis

## Goal

Treat a trained or constructed spiking network as an experimental system that can be perturbed.

## Experiments

Perturb the network by:

- injecting current into selected populations
- silencing neurons
- changing synaptic weights
- temporarily altering inhibition
- displacing population state
- adding structured noise

Apply perturbations at different points during a task.

## Analysis

Measure:

- recovery time
- trajectory displacement
- output changes
- transitions between attractors
- sensitivity to perturbation direction
- sensitivity to perturbation timing

## Key questions

- Which dimensions of neural activity matter for computation?
- Which perturbations are rapidly corrected?
- Which perturbations permanently change network state?

This project should help connect dynamical-systems reasoning with the logic of perturbation experiments in neuroscience.

---

# Project 23 — Decode Hidden Network State from Spikes

## Goal

Reverse the perspective.

Instead of inspecting known internal variables directly, treat spike trains as experimental observations.

## Setup

Use a network from an earlier project with a known latent variable such as:

- attractor identity
- represented angle
- accumulated evidence
- stimulus history
- task context
- reservoir state

Hide that variable from the analysis pipeline.

## Task

Infer it from population spikes.

Try several representations:

- spike counts in windows
- filtered firing rates
- population vectors
- low-dimensional latent representations

## Analysis

Compare decoding methods.

Study:

- number of observed neurons
- observation window length
- noise
- temporal resolution
- population subsampling

## Key question

How much information about the underlying dynamical state is actually observable from finite spike data?

---

# Project 24 — Connectivity Inference from Spiking Activity

## Goal

Investigate how much network structure can be recovered from observed spike trains.

## Setup

Generate data from a recurrent network whose connectivity is completely known.

Then hide the connectivity matrix from the inference procedure.

## Begin simply

Try basic statistical relationships such as:

- correlations
- lagged correlations
- cross-correlograms

Demonstrate why these are not equivalent to connectivity.

## Progress to model-based approaches

Explore an appropriate statistical model for predicting spikes from:

- stimulus history
- self-history
- other neurons' histories

A point-process GLM is a natural candidate.

## Evaluation

Because the ground truth network is known, evaluate:

- true positive connections
- false positives
- inferred sign
- weight estimates
- dependence on firing rate
- dependence on recording duration
- dependence on hidden neurons

## Key questions

- When does functional interaction resemble anatomical connectivity?
- When does common input create false connections?
- How badly do unobserved neurons complicate inference?

---

# Project 25 — System Identification of a Hidden Spiking Network

## Goal

Combine the previous inference ideas into a small system-identification challenge.

## Setup

Create a recurrent spiking system with known dynamics.

Generate synthetic "experimental" data under multiple input conditions.

Then treat the simulator as hidden.

The analysis should have access only to:

- inputs
- observed spike trains

## Tasks

Attempt to recover:

- relevant state variables
- effective time scales
- input-output relationships
- dynamical regime
- low-dimensional latent structure
- possibly effective interactions between neurons

## Validation

Compare every inferred quantity against simulator ground truth.

## Key question

Which properties of a recurrent neural system are identifiable from its observable activity, and which are fundamentally ambiguous?

---

# Project 26 — Model Recovery and Identifiability

## Goal

Study inference failure rather than only inference success.

## Setup

Construct several networks with different mechanisms that produce superficially similar neural activity.

Examples might include:

- recurrent persistent activity versus slowly decaying synapses
- common input versus recurrent coupling
- oscillatory input versus internally generated oscillation
- different network structures producing similar firing-rate statistics

## Task

Attempt to distinguish the models from observations.

Determine what additional experiments or perturbations would make them distinguishable.

## Key concepts

Explore:

- identifiability
- model degeneracy
- observational equivalence
- experimental design

## Key question

If two models explain passive observations equally well, what intervention would discriminate between them?

---

# Project 27 — Emergent Dynamics in Larger E/I Networks

## Goal

Scale from small interpretable circuits toward population-level recurrent dynamics.

## Network

Construct a substantially larger sparse E/I spiking network.

Explore regimes including:

- asynchronous irregular activity
- oscillatory activity
- strongly synchronized activity
- metastable activity
- potentially chaotic or highly sensitive dynamics

## Analysis

Characterize:

- firing-rate distributions
- pairwise correlations
- population dimensionality
- temporal autocorrelation
- spectral structure
- trial-to-trial variability
- sensitivity to perturbations

## Key question

Which macroscopic dynamical properties remain predictable despite complicated microscopic spike activity?

---

# Project 28 — Metastable Population Dynamics

## Goal

Study networks that spontaneously transition among quasi-stable population states.

## Network

Construct or tune a recurrent network with several transiently stable activity configurations.

These states should persist for some time but eventually transition.

## Analysis

Try to identify the hidden states from activity without using ground-truth labels.

Explore:

- clustering
- dimensionality reduction
- state-transition matrices
- dwell-time distributions
- transition probabilities

## Key questions

- How should a metastable neural state be defined?
- Can states be reliably inferred from spikes?
- Are transitions noise-driven or structurally determined?
- How does metastability differ from true attractor dynamics?

---

# Project 29 — Computational Primitive Library

## Goal

Revisit the previous projects from the perspective of computation.

Rather than treating networks primarily as biological models, ask what operations recurrent spiking dynamics naturally implement.

Construct small demonstrations of primitives such as:

- integration
- leaky integration
- thresholding
- comparison
- winner-take-all
- normalization
- memory
- switching
- gating
- routing
- oscillation
- sequence generation
- continuous-variable storage

For each primitive identify:

1. the input representation
2. the output representation
3. the relevant state variables
4. the dynamical mechanism performing the computation
5. robustness to noise
6. characteristic time scale
7. whether recurrence is essential

## Important goal

Develop a vocabulary connecting:

**mathematical operation → dynamical mechanism → spiking implementation**

This project should synthesize lessons from the entire sequence.

---

# Project 30 — Composite Spiking Computation

## Goal

Combine several computational primitives into a larger functional system.

Possible examples include:

- gated evidence accumulator
- context-dependent working memory
- memory followed by comparison
- sequence-controlled routing
- attractor memory feeding a decision circuit

The particular system can be chosen based on what was most interesting in earlier projects.

## Analysis

Identify which subnetworks perform which dynamical operations.

Perturb each component independently and determine how the overall computation changes.

## Key question

Can a moderately complex computation be understood compositionally in terms of simpler recurrent dynamical primitives?

---

# Project 31 — Fit a Recurrent Spiking Model to Synthetic Population Data

## Goal

Perform genuine network-level parameter fitting.

## Setup

Generate synthetic data from a recurrent SNN.

Hide some of the simulator parameters.

Candidate parameters might include:

- recurrent coupling strengths
- synaptic time constants
- input strengths
- noise levels
- connectivity statistics
- E/I balance

## Task

Fit the model using observable population statistics.

Possible target statistics include:

- firing-rate distributions
- autocorrelation
- cross-correlation
- population-rate spectra
- response to stimuli
- latent trajectory geometry

## Important issue

Investigate whether matching summary statistics uniquely constrains the network.

## Analysis

Compare:

- true parameters
- recovered parameters
- uncertainty
- degeneracies between parameters

## Key question

Can a network reproduce the data for the wrong mechanistic reasons?

---

# Project 32 — Fit Population Dynamics Rather Than Individual Spikes

## Goal

Explore fitting at the level of collective dynamics.

## Setup

Generate repeated trials from a recurrent spiking system.

Instead of trying to reproduce exact spike trains, fit network parameters to reproduce population-level behavior.

Potential targets include:

- mean trajectories
- covariance
- latent dynamics
- dimensionality
- attractor locations
- transition probabilities
- response to perturbations

## Compare

Contrast fitting based on:

- firing rates alone
- pairwise statistics
- temporal statistics
- latent population trajectories
- perturbation responses

## Key question

Which observables are most informative about the underlying recurrent dynamics?

---

# Project 33 — Real Neural Population Data

## Goal

Apply the analysis and modeling tools developed above to an actual neural dataset.

Choose a dataset containing simultaneous recordings from many neurons during a reasonably interpretable behavioral or sensory task.

## Initial analysis

Characterize:

- firing rates
- trial structure
- task variables
- population dimensionality
- condition-dependent trajectories
- decoding performance
- temporal structure

## Modeling

Construct a recurrent spiking model intended to reproduce selected features of the data.

The goal is **not** necessarily to reproduce every spike.

Instead identify a specific set of phenomena to explain, such as:

- persistent activity
- tuning
- population trajectories
- choice-related signals
- temporal integration
- sequence structure

## Validation

Compare model and data using multiple statistics rather than a single loss.

## Key question

Which experimentally observed properties require recurrent dynamical structure, and which can be reproduced trivially?

---

# Project 34 — Model Comparison on Real Data

## Goal

Move from "can a model reproduce the data?" to "which mechanistic hypotheses are actually supported?"

## Setup

Construct several competing recurrent SNN models representing different mechanisms for the same observed phenomenon.

For example:

- persistent attractor
- transient trajectory
- feedforward sequence
- slowly decaying input
- reservoir-like recurrent dynamics

## Compare models using

- held-out predictive performance
- population statistics
- latent dynamics
- perturbation predictions where possible
- model complexity
- parameter identifiability

## Key question

What experimental evidence genuinely distinguishes different dynamical explanations of the same neural activity?

---

# Capstone — Build, Observe, Infer, and Perturb a Neural Dynamical System

## Goal

Integrate the entire sequence into one end-to-end study.

## Phase 1 — Construct

Build a recurrent spiking network that performs a nontrivial temporal computation.

The network should have:

- recurrent dynamics
- internal state
- a meaningful task
- multiple experimental conditions

## Phase 2 — Characterize

Analyze the known network using full simulator access.

Identify:

- dynamical variables
- attractors or trajectories
- relevant timescales
- population geometry
- computation performed by the network

## Phase 3 — Create synthetic experiments

Generate trials as though the network were a biological system.

Record only a subset of neurons.

Add realistic limitations such as:

- finite trials
- noise
- partial observability
- variable initial states

## Phase 4 — Infer

Without using privileged simulator information, attempt to infer:

- task variables
- latent state
- effective dynamics
- interactions
- relevant timescales

## Phase 5 — Perturb

Design interventions based on the inferred model.

Predict the consequences before running the simulations.

Then perform the perturbations on the ground-truth network.

## Phase 6 — Evaluate

Compare:

- inferred mechanism
- true mechanism
- predicted perturbation response
- actual perturbation response

## Final question

How much understanding of a neural dynamical system can be obtained from observation alone, and how much requires intervention?

---

# Notebook Expectations

Each notebook should generally contain the following conceptual elements, although their exact organization is flexible.

## Motivation

Briefly state:

- what phenomenon is being studied
- why it matters
- what dynamical or computational idea is being investigated

Avoid long textbook introductions.

## Mathematical model

State the important equations and assumptions.

The reader should be able to understand what system is being simulated without reverse-engineering the code.

## Experiment

Clearly define:

- inputs
- network
- manipulation
- measured outputs

## Visualization

Prefer plots that answer a scientific question.

Avoid producing plots merely because they are easy to generate.

## Analysis

Go beyond raster plots.

Whenever appropriate, analyze population activity as a dynamical system.

## Interpretation

End with a concise discussion of:

- what happened
- why it happened
- what the experiment demonstrates
- what remains ambiguous

## Exercises / experiments

End each notebook with several suggested modifications for further exploration.

These should usually involve changing the science rather than rewriting the software.

---

# Pedagogical Style

The notebooks should assume a technically sophisticated reader who is new to recurrent SNN modeling but not new to computational science.

Therefore:

- do not spend excessive time explaining Python
- do not explain basic differential equations
- do not re-teach basic neuron models
- do not provide generic introductions to machine learning
- do explain unfamiliar neuroscience and dynamical-systems concepts when they become relevant
- connect equations, simulations, and observed behavior
- emphasize interpretation over library syntax

Terminology should be precise but plain.

Avoid unnecessary software jargon.

---

# Relationship Between Projects

The projects should reuse ideas from previous notebooks.

For example:

- the bistable circuit becomes the first attractor model
- attractor analysis motivates state-space analysis
- state-space analysis becomes useful for trained recurrent networks
- reservoir analysis introduces decoding hidden network state
- decoding leads naturally to system identification
- system identification motivates identifiability and perturbation experiments
- those ideas culminate in fitting and testing models against real population data

The sequence should therefore feel like one developing investigation rather than a collection of unrelated tutorials.

---

# Core Themes to Reinforce

Across the entire project sequence, repeatedly return to these ideas:

### Spikes versus state

Spikes are observable events, but the computation often becomes clearer when expressed in terms of population-level dynamical variables.

### Recurrence creates internal state

A recurrent network can contain information about its past even after the original input disappears.

### Computation is dynamics

Integration, memory, switching, decision-making, sequence generation, and routing can all be viewed as trajectories through state space.

### Similar observations can arise from different mechanisms

Matching firing rates or raster plots does not imply that the underlying model is correct.

### Partial observability matters

Experimental recordings reveal only a fraction of the underlying neural system.

### Perturbations are especially informative

Observational equivalence can often be broken by intervening on the system.

### Modeling and inference are dual problems

One direction asks:

> Given a network, what activity does it produce?

The reverse asks:

> Given activity, what can we infer about the network and its dynamics?

Both perspectives should be developed throughout the projects.

---

# Overall Progression

The intended progression is approximately:

**recurrent interaction**

→ **population state**

→ **bistability**

→ **attractors**

→ **continuous representations**

→ **temporal integration**

→ **reservoir dynamics**

→ **trained recurrent SNNs**

→ **population geometry**

→ **perturbation**

→ **decoding**

→ **connectivity inference**

→ **system identification**

→ **identifiability**

→ **large-network emergent dynamics**

→ **computational primitives**

→ **composite computation**

→ **network fitting**

→ **real neural population data**

The end goal is not simply proficiency with an SNN simulator.

The goal is to develop intuition for **how recurrent spiking systems compute, how their computation appears in population activity, and how one can infer the underlying dynamics from limited observations.**