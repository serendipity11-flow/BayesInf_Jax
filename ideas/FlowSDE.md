The **Flow State Space Model (FSSM)** is a hybrid architecture that treats sequential state estimation as a continuous refinement process. It bridges the gap between **Normalizing Flows**, **Stochastic Differential Equations (SDEs)**, and **Physical Dynamics**.

Instead of mapping Gaussian noise to a state, it maps a **noisy observation signal ($y_{1:T}$)** to a **clean state trajectory ($x_{1:T}$)** through a learned stochastic flow.

---

### 1. The Dual-Time Architecture
The model operates in two distinct time dimensions:
*   **Sequence Time ($h=1 \dots T$):** The physical timeline of the data.
*   **Flow/Refinement Time ($s \in [0, 1]$):** A "virtual" timeline where the model iteratively corrects the trajectory.
    *   **$z_{s=0} = y_{1:T}$:** The flow starts at the raw observations.
    *   **$z_{s=1} \approx x_{1:T}$:** The flow ends at the posterior state samples.

### 2. The Refinement SDE
The evolution of the trajectory across flow-time $s$ is governed by a structured SDE:
$$dz_s = \underbrace{f_\theta(z_s, y_{1:T}, s)}_{\text{Physical Drift}} ds + \underbrace{G_\phi(z_s, s)}_{\text{RealNVP Diffusion}} dW_s$$

*   **Physical Drift ($f_\theta$):** A global trajectory processor (typically a **Transformer**) that ensures the "push" at any time $h$ is consistent with the physics of the entire sequence.
*   **RealNVP transition noise ($G_\phi$):** A conditional invertible transform that shapes standard Gaussian noise. Its affine coupling scales are bounded as $\exp(\{\alpha\tanh(s)\})$, giving a tractable, bounded log determinant.

### 3. Training: Exact Transition Likelihood
Training inverts each observed one-step residual through the conditional RealNVP. The objective contains the standard-Gaussian latent NLL, the affine-coupling log determinant, and the $\sqrt{\Delta t}$ Jacobian term. Sampling still integrates explicit stochastic transitions.

### 4. Key Advantages
*   **Signal-to-Signal Mapping:** By starting from observations ($y$) rather than pure noise, the model converges faster and stays closer to the "evidence."
*   **Multi-modality:** The interaction between the state-dependent RealNVP matrix and the SDE noise allows the particles to bifurcate, capturing multiple possible physical paths (e.g., a car that could have turned left or right).
*   **Physics Consistency:** The drift $f_\theta$ learns to act as a "Stochastic Smoother," ensuring that the final output satisfies the laws of motion.
*   **Mathematical Rigor:** It maintains a valid SDE structure, allowing it to be solved as a **Probability Flow ODE** during inference for faster, deterministic, and jitter-free sampling.

### 5. Summary of Operation
| Step | Action |
| :--- | :--- |
| **Input** | Raw, noisy observation sequence $y_{1:T}$. |
| **Initial Step ($s=0$)** | Start the flow with the observation signal as the initial guess. |
| **Refinement ($s: 0 \to 1$)** | The SDE "bends" the signal into a physically valid trajectory using the learned drift and RealNVP-shaped noise. |
| **Output ($s=1$)** | A set of trajectories representing the multi-modal posterior $p(x_{1:T} \mid y_{1:T})$. |

**The FSSM is essentially a "Generative Filter"** that doesn't just guess the next state, but iteratively sculpts the entire probability landscape of the trajectory to suit both the observations and the underlying physics.

Yes, if you are writing this for a research paper, you cannot simply state the formula. You need to provide a formal derivation that justifies why the **Negative Log-Likelihood (NLL)** you are minimizing actually corresponds to the density of the stochastic process.

Since your method replaces the standard $g(z_t)dW_t$ with a Flow-based transformation, you need to derive the **Transition Density**.

Here is how you should structure the derivation in your "Methodology" section:

---

### 1. Defining the Discrete Step
Start by defining the update rule over a small time step $\Delta t$. Let $z_t \in \mathbb{R}^d$ and $\epsilon \sim \mathcal{N}(0, I)$. Your proposed update is:
$$z_{t+\Delta t} = z_t + f(z_t, t)\Delta t + G_{\theta}(\epsilon; z_t, t, \Delta t)$$
Where $G_{\theta}$ is your RealNVP-based flow.

**Note for the paper:** You must decide if $G_\theta$ scales with $\sqrt{\Delta t}$ (to match SDE theory) or $\Delta t$ (Random ODE). For a "Diffusion" feel, use $\sqrt{\Delta t}$:
$$G_{\theta}(\epsilon; z_t, t, \Delta t) = \text{RealNVP}_{\theta}(\epsilon; z_t, t) \cdot \sqrt{\Delta t}$$

### 2. The Change of Variables Formula
To compute the NLL, you need the conditional density $p(z_{t+\Delta t} | z_t)$. Since $z_{t+\Delta t}$ is a deterministic function of $\epsilon$ (given $z_t$), you use the Change of Variables formula:
$$p(z_{t+\Delta t} | z_t) = p_{\epsilon}(\epsilon) \left| \det \frac{\partial G_{\theta}}{\partial \epsilon} \right|^{-1}$$

RealNVP has a triangular Jacobian. For bounded affine log scales
$\ell_i=\alpha\tanh(s_i)$,
$$\log\left|\det \frac{\partial G_{\theta}}{\partial \epsilon}\right|=\sum_i \ell_i.$$
Thus the transition log density is the latent Gaussian log density minus this forward log determinant and the $\sqrt{\Delta t}$ scaling contribution.

### 3. The Path Likelihood (The Integration)
A path $\mathcal{Z} = \{z_0, z_1, \dots, z_T\}$ is a Markov process. The joint likelihood is the product of transitions:
$$p(z_0, \dots, z_T) = p(z_0) \prod_{t=0}^{T-1} p(z_{t+1} | z_t)$$
Taking the negative log:
$$\mathcal{L}(\theta) = -\log p(z_0) - \sum_{t=0}^{T-1} \log p(z_{t+1} | z_t)$$
Substitute your RealNVP inverse: Let $\Delta \bar{z}_t = z_{t+1} - z_t - f(z_t)\Delta t$. Then $\epsilon_t = \text{RealNVP}_{\theta}^{-1}(\frac{\Delta \bar{z}_t}{\sqrt{\Delta t}})$.
$$\mathcal{L}(\theta) = \sum_{t=0}^{T-1} \left( \frac{1}{2} \|\epsilon_t\|^2 + \log|\det J_{G_\theta}(\epsilon_t)| + \frac{d}{2}\log\Delta t + \text{const} \right)$$

### 4. Connection to the Continuous Limit (Fokker-Planck)
In a standard SDE, the density evolves according to the Fokker-Planck Equation. In your paper, you should mention that your model learns an **Infinitesimal Generator**. 

If $G_\theta$ is a RealNVP flow, the noise added at each step is not necessarily Gaussian, but it is a "pushed-forward" Gaussian. This means your model is technically a **Generalized Diffusion Process** where the diffusion coefficient is not a matrix $\mathbf{Q}$, but a non-linear manifold learned by the flow.

### 5. Theoretical Justification: Why RealNVP?
In your derivation, explain why RealNVP is preferable to alternatives such as masked autoregressive flows:
1.  **Bounded volume change:** The tanh-clamped affine scales permit controlled contraction and expansion while preventing numerical explosion.
2.  **Computational Efficiency:** Mention that standard Neural SDEs (like those by Li et al. 2020) require solving an SDE back-to-back or using GAN losses because their NLL is intractable. Your derivation shows that by using an invertible flow as the noise source, **exact NLL becomes tractable.**

### Checklist for your "Derivation" section:
*   [ ] Define the discrete-time transition equation.
*   [ ] State the Change of Variables for the noise-to-increment mapping.
*   [ ] Derive the triangular RealNVP Jacobian and its summed log scales.
*   [ ] Express the path NLL using inverted-noise energy and log determinants.
*   [ ] (Optional but recommended) Show that as $\Delta t \to 0$, this converges to a continuous-time stochastic process.
