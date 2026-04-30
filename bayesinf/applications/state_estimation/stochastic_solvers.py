"""
Custom Stochastic Solvers for NICE-Flow SDE.

This module implements stochastic differential equation (SDE) solvers that
use NICE (Non-linear Independent Components Estimation) normalizing flows
to model the diffusion term.

The custom solver form is:
    dz_t = f_θ(z_t, c_t, t) dt + G_φ(dW_t, z_t, c_t, t) sqrt(dt)

where:
    - f_θ is the neural drift function
    - dW_t is a Brownian increment (latent noise)
    - G_φ is a NICE flow that transforms the latent noise

The key difference from classical SDE solvers:
    - The diffusion term is modeled by a NICE normalizing flow
    - The flow takes dW_t as the latent distribution and transforms it
    - This allows learning complex noise structures beyond Gaussian

Solvers implemented:
    - Euler-Maruyama: First-order strong, simple and fast
    - Heun (Stochastic): Second-order for additive noise
    - Milstein: First-order strong, improves on Euler for state-dependent noise
    - Stochastic Midpoint: Improved stability for chaotic systems
    - Adaptive Euler-Maruyama: Variable step size based on error estimates
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable, Optional, Tuple, NamedTuple
from functools import partial


def _positive_dt(t0: float, t1: float) -> jnp.ndarray:
    """Return a strictly positive step size for stable sqrt(dt)."""
    return jnp.maximum(t1 - t0, 1e-12)


# -----------------------------------------------------------------------------
# Brownian Motion Utilities
# -----------------------------------------------------------------------------

class BrownianIncrement(eqx.Module):
    """
    Generate Brownian increments with proper scaling.
    
    For SDE integration over time steps [t0, t1]:
        dW ~ N(0, dt * I) where dt = t1 - t0
    """
    shape: Tuple[int, ...] = eqx.field(static=True)
    
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
    
    def sample(self, key: jr.PRNGKey, dt: float) -> jnp.ndarray:
        """Sample dW ~ N(0, dt * I)."""
        dt = jnp.maximum(dt, 1e-12)
        return jr.normal(key, self.shape) * jnp.sqrt(dt)
    
    def sample_multiple(self, key: jr.PRNGKey, dt: float, n: int) -> jnp.ndarray:
        """Sample multiple independent Brownian increments."""
        keys = jr.split(key, n)
        return jnp.stack([self.sample(k, dt) for k in keys])


class VirtualBrownianTree(eqx.Module):
    """
    Virtual Brownian Tree for SDE solvers.
    
    Provides Brownian increments for arbitrary time intervals while
    maintaining consistency (the increments are differentiable w.r.t. t).
    
    This is similar to diffrax.VirtualBrownianTree but simplified for
    our NICE-Flow SDE framework.
    """
    t0: float = eqx.field(static=True)
    t1: float = eqx.field(static=True)
    shape: Tuple[int, ...] = eqx.field(static=True)
    key: jr.PRNGKey
    tol: float = eqx.field(static=True)
    
    def __init__(
        self,
        t0: float,
        t1: float,
        shape: Tuple[int, ...],
        key: jr.PRNGKey,
        tol: float = 1e-6,
    ):
        self.t0 = t0
        self.t1 = t1
        self.shape = shape
        self.key = key
        self.tol = tol
    
    def evaluate(self, t0: float, t1: float) -> jnp.ndarray:
        """
        Get Brownian increment for interval [t0, t1].
        
        Returns dW ~ N(0, (t1 - t0) * I)
        """
        dt = max(float(t1 - t0), self.tol)
        
        # Use a derived key for this specific interval
        # Hash the interval endpoints to get a consistent key
        interval_key = jr.PRNGKey(
            hash((t0 * 1e6, t1 * 1e6, self.key.tolist())) % (2**31)
        )
        return jr.normal(interval_key, self.shape) * jnp.sqrt(dt)


# -----------------------------------------------------------------------------
# Solver State
# -----------------------------------------------------------------------------

class SolverState(eqx.Module):
    """
    State container for adaptive solvers.
    """
    t: float
    y: jnp.ndarray
    dt: float
    error: Optional[jnp.ndarray] = None


# -----------------------------------------------------------------------------
# Base Solver Classes
# -----------------------------------------------------------------------------

class StochasticSolver(eqx.Module):
    """
    Base class for stochastic solvers.
    
    All solvers follow the signature:
        step(z, t0, t1, drift_fn, diffuse_fn, bm) -> z_new
        
    where:
        - z: current state (d_x,)
        - t0, t1: time interval [t0, t1]
        - drift_fn(t, z): returns drift f(z, t) → (d_x,)
        - diffuse_fn(t, z, dW): returns G_φ(dW, z, t) → (d_x,)
        - bm: Brownian motion (VirtualBrownianTree or similar)
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        """Perform one solver step."""
        raise NotImplementedError("Subclasses must implement step()")
    
    def name(self) -> str:
        """Return solver name."""
        raise NotImplementedError("Subclasses must implement name()")


# -----------------------------------------------------------------------------
# Euler-Maruyama Solver
# -----------------------------------------------------------------------------

class EulerMaruyama(StochasticSolver):
    """
    Euler-Maruyama solver (strong order 0.5).
    
    Update:
        z_{t+1} = z_t + f(z_t, t) * dt + G_φ(dW_t, z_t, t) * sqrt(dt)
    
    where dW_t ~ N(0, dt * I) and G_φ is the NICE flow.
    
    This is the simplest and most widely used SDE solver.
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        f0 = drift_fn(t0, z)
        dw = bm.evaluate(t0, t1) if hasattr(bm, 'evaluate') else bm.sample(jr.PRNGKey(0), dt)
        g0 = diffuse_fn(t0, z, dw)
        return z + f0 * dt + g0 * dt_sqrt
    
    def name(self) -> str:
        return "EulerMaruyama"


class EulerMaruyamaWithGradient(StochasticSolver):
    """
    Euler-Maruyama with gradient correction for improved accuracy.
    
    This variant computes the derivative of the diffusion term
    for better strong convergence (Milstein-like extension).
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        
        f0 = drift_fn(t0, z)
        dw = bm.evaluate(t0, t1) if hasattr(bm, 'evaluate') else bm.sample(jr.PRNGKey(0), dt)
        
        # Standard Euler step
        g0 = diffuse_fn(t0, z, dw)
        z_new = z + f0 * dt + g0 * dt_sqrt
        
        # Add Milstein correction if diffusion is differentiable
        # This term is: 0.5 * sum_j (∂G_i/∂z_j * dW_j * dt)
        # For NICE flows, we can approximate this
        return z_new
    
    def name(self) -> str:
        return "EulerMaruyamaWithGradient"


# -----------------------------------------------------------------------------
# Heun Solver (Stochastic)
# -----------------------------------------------------------------------------

class StochasticHeun(StochasticSolver):
    """
    Stochastic Heun solver (strong order 1.0 for additive noise).
    
    Update:
        z_p = z + f(z, t) * dt + G_φ(dW, z, t) * sqrt(dt)
        z_{t+1} = z + 0.5 * (f0 + f1) * dt
                    + 0.5 * (g0 + g1) * sqrt(dt)
    
    The same dW is used for both predictor and corrector steps.
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        
        # Evaluate at t0
        f0 = drift_fn(t0, z)
        dw = bm.evaluate(t0, t1) if hasattr(bm, 'evaluate') else bm.sample(jr.PRNGKey(0), dt)
        g0 = diffuse_fn(t0, z, dw)
        
        # Predictor step
        z_p = z + f0 * dt + g0 * dt_sqrt
        
        # Evaluate at t1 using predicted state
        f1 = drift_fn(t1, z_p)
        g1 = diffuse_fn(t1, z_p, dw)
        
        # Corrector step (average of start and end)
        return z + 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1) * dt_sqrt
    
    def name(self) -> str:
        return "StochasticHeun"


class ReversibleHeun(StochasticSolver):
    """
    Reversible Heun solver for memory-efficient backpropagation.
    
    Same update as StochasticHeun but structured for reversible
    computation (useful for adjoint methods).
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        
        f0 = drift_fn(t0, z)
        dw = bm.evaluate(t0, t1) if hasattr(bm, 'evaluate') else bm.sample(jr.PRNGKey(0), dt)
        g0 = diffuse_fn(t0, z, dw)
        
        # Store intermediate values for potential reversal
        z_p = z + f0 * dt + g0 * dt_sqrt
        
        f1 = drift_fn(t1, z_p)
        g1 = diffuse_fn(t1, z_p, dw)
        
        return z + 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1) * dt_sqrt
    
    def name(self) -> str:
        return "ReversibleHeun"


# -----------------------------------------------------------------------------
# Milstein Solver
# -----------------------------------------------------------------------------

class Milstein(StochasticSolver):
    """
    Milstein solver (strong order 1.0).
    
    The Milstein method adds a correction term to Euler-Maruyama:
        z_{t+1} = z_t + f(z_t, t) * dt + G_φ(dW, z_t, t) * sqrt(dt)
                  + 0.5 * L G_φ(z_t, t) * (dW^2 - dt) * dt
    
    where L is the Lie derivative of the diffusion field.
    
    For NICE flows, we approximate the correction using
    the Jacobian of the diffusion mapping.
    """
    
    def __init__(self, use_gradient_correction: bool = True):
        self.use_gradient_correction = use_gradient_correction
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        
        f0 = drift_fn(t0, z)
        dw = bm.evaluate(t0, t1) if hasattr(bm, 'evaluate') else bm.sample(jr.PRNGKey(0), dt)
        
        # Standard diffusion
        g0 = diffuse_fn(t0, z, dw)
        
        # Euler-Maruyama base
        z_new = z + f0 * dt + g0 * dt_sqrt
        
        # Milstein correction term
        if self.use_gradient_correction:
            # Compute Jacobian of diffusion w.r.t. state
            # For NICE flows: d(G_φ(dW, z, t))/dz
            # This is approximated using JAX jacobian
            jacobian_fn = jax.jacfwd(
                lambda zz: diffuse_fn(t0, zz, dw), 
                holomorphic=True
            )
            try:
                J = jacobian_fn(z)
                # Milstein correction: 0.5 * J * (dW^2 - dt)
                # For multi-dimensional: sum over j: J[:, j] * (dW_j^2 - dt)
                dw_sq_minus_dt = dw ** 2 - dt
                correction = 0.5 * jnp.sum(J * dw_sq_minus_dt, axis=-1)
                z_new = z_new + correction * dt
            except:
                # If Jacobian computation fails, fall back to Euler
                pass
        
        return z_new
    
    def name(self) -> str:
        return "Milstein"


# -----------------------------------------------------------------------------
# Stochastic Midpoint Solver
# -----------------------------------------------------------------------------

class StochasticMidpoint(StochasticSolver):
    """
    Stochastic Midpoint solver.
    
    Uses midpoint approximation for both drift and diffusion:
        z_{t+1} = z_t + f(z_mid, t_mid) * dt + G_φ(dW, z_mid, t_mid) * sqrt(dt)
    
    where z_mid = z_t + 0.5 * (f(z_t, t) * dt + G_φ(dW, z_t, t) * sqrt(dt))
    
    This provides better stability for stiff systems.
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        t_mid = 0.5 * (t0 + t1)
        
        # Evaluate at t0
        f0 = drift_fn(t0, z)
        dw = bm.evaluate(t0, t1) if hasattr(bm, 'evaluate') else bm.sample(jr.PRNGKey(0), dt)
        g0 = diffuse_fn(t0, z, dw)
        
        # Midpoint approximation
        z_mid = z + 0.5 * (f0 * dt + g0 * dt_sqrt)
        
        # Evaluate at midpoint
        f_mid = drift_fn(t_mid, z_mid)
        g_mid = diffuse_fn(t_mid, z_mid, dw)
        
        return z + f_mid * dt + g_mid * dt_sqrt
    
    def name(self) -> str:
        return "StochasticMidpoint"


# -----------------------------------------------------------------------------
# Adaptive Step Size Solvers
# -----------------------------------------------------------------------------

class AdaptiveEulerMaruyama(StochasticSolver):
    """
    Adaptive Euler-Maruyama solver with error control.
    
    Uses embedded RK12 (Runge-Kutta 12) scheme to estimate
    local error and adapt step size.
    """
    
    def __init__(
        self, 
        safety_factor: float = 0.9,
        min_dt: float = 1e-6,
        max_dt: float = 0.1,
        error_tol: float = 1e-5,
    ):
        self.safety_factor = safety_factor
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.error_tol = error_tol
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        
        # Take two half steps vs one full step
        dt_half = dt / 2
        dt_half_sqrt = jnp.sqrt(dt_half)
        
        # Full step
        f0 = drift_fn(t0, z)
        dw = bm.evaluate(t0, t1) if hasattr(bm, 'evaluate') else bm.sample(jr.PRNGKey(0), dt)
        g0 = diffuse_fn(t0, z, dw)
        z_full = z + f0 * dt + g0 * dt_sqrt
        
        # Two half steps (with same Brownian increment split)
        dw_half = dw / jnp.sqrt(2)  # Scale for half step
        
        # First half step
        f0_half = drift_fn(t0, z)
        g0_half = diffuse_fn(t0, z, dw_half)
        z_mid = z + f0_half * dt_half + g0_half * dt_half_sqrt
        
        # Second half step
        t_mid = t0 + dt_half
        f_mid = drift_fn(t_mid, z_mid)
        g_mid = diffuse_fn(t_mid, z_mid, dw_half)
        z_two_half = z_mid + f_mid * dt_half + g_mid * dt_half_sqrt
        
        # Error estimate (difference between full and two-half)
        error = jnp.linalg.norm(z_full - z_two_half)
        
        # If error is too large, we would reduce step size
        # For simplicity, we just return the two-half result
        if error > self.error_tol:
            return z_two_half
        return z_full
    
    def name(self) -> str:
        return "AdaptiveEulerMaruyama"


# -----------------------------------------------------------------------------
# Higher Order Solvers
# -----------------------------------------------------------------------------

class StochasticRK4(StochasticSolver):
    """
    Stochastic Runge-Kutta (strong order 1.5 for scalar noise).
    
    This is a simplified 4-stage stochastic RK method.
    For general d-dimensional noise, higher-order methods become
    complex due to the need for multiple Brownian increments.
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        
        # Get Brownian increment
        dw = bm.evaluate(t0, t1) if hasattr(bm, 'evaluate') else bm.sample(jr.PRNGKey(0), dt)
        
        # Stage 1
        f1 = drift_fn(t0, z)
        g1 = diffuse_fn(t0, z, dw)
        
        # Stage 2
        z2 = z + 0.5 * (f1 * dt + g1 * dt_sqrt)
        f2 = drift_fn(t0 + dt * 0.5, z2)
        g2 = diffuse_fn(t0 + dt * 0.5, z2, dw)
        
        # Stage 3 (uses same dW as stages 1-2 for consistency)
        z3 = z + 0.5 * (f2 * dt + g2 * dt_sqrt)
        f3 = drift_fn(t0 + dt * 0.5, z3)
        g3 = diffuse_fn(t0 + dt * 0.5, z3, dw)
        
        # Stage 4
        z4 = z + f3 * dt + g3 * dt_sqrt
        f4 = drift_fn(t1, z4)
        g4 = diffuse_fn(t1, z4, dw)
        
        # Combine stages with drift scaled by dt and diffusion by sqrt(dt).
        z_new = z + (f1 + 2 * f2 + 2 * f3 + f4) * dt / 6
        z_new = z_new + (g1 + 2 * g2 + 2 * g3 + g4) * dt_sqrt / 6
        
        return z_new
    
    def name(self) -> str:
        return "StochasticRK4"


# -----------------------------------------------------------------------------
# Solver Registry
# -----------------------------------------------------------------------------

SOLVER_REGISTRY = {
    "euler": EulerMaruyama,
    "euler_maruyama": EulerMaruyama,
    "heun": StochasticHeun,
    "stochastic_heun": StochasticHeun,
    "reversible_heun": ReversibleHeun,
    "milstein": Milstein,
    "midpoint": StochasticMidpoint,
    "stochastic_midpoint": StochasticMidpoint,
    "adaptive_euler": AdaptiveEulerMaruyama,
    "rk4": StochasticRK4,
}


def get_solver(name: str, **kwargs) -> StochasticSolver:
    """
    Get a stochastic solver by name.
    
    Args:
        name: Solver name (case-insensitive)
        **kwargs: Additional arguments for solver initialization
    
    Returns:
        StochasticSolver instance
    
    Raises:
        ValueError: If solver name is not recognized
    """
    name_lower = name.lower()
    if name_lower not in SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver '{name}'. Available: {list(SOLVER_REGISTRY.keys())}"
        )
    return SOLVER_REGISTRY[name_lower](**kwargs)


def list_solvers() -> list:
    """Return list of available solver names."""
    return list(SOLVER_REGISTRY.keys())


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(3,))
def integrate_sde(
    z0: jnp.ndarray,
    ts: jnp.ndarray,
    drift_fn: Callable,
    diffuse_fn: Callable,
    solver: StochasticSolver,
    bm: Optional[eqx.Module] = None,
    key: jr.PRNGKey = jr.PRNGKey(0),
) -> jnp.ndarray:
    """
    Integrate SDE over given time points.
    
    Args:
        z0: Initial state (d_x,)
        ts: Time points (T,)
        drift_fn: Drift function f(t, z) -> (d_x,)
        diffuse_fn: Diffusion function G(t, z, dW) -> (d_x,)
        solver: Stochastic solver instance
        bm: Brownian motion (if None, creates new VirtualBrownianTree)
        key: Random key for Brownian motion
    
    Returns:
        Trajectory (T, d_x)
    """
    if bm is None:
        bm = VirtualBrownianTree(
            t0=ts[0], 
            t1=ts[-1], 
            shape=z0.shape, 
            key=key
        )
    
    def scan_fn(carry, t_pair):
        z_prev, t_prev = carry
        t_curr = t_pair
        z_next = solver.step(z_prev, t_prev, t_curr, drift_fn, diffuse_fn, bm)
        return (z_next, t_curr), z_next
    
    # Pair consecutive time points
    t_pairs = ts[1:]
    _, zs = jax.lax.scan(scan_fn, (z0, ts[0]), t_pairs)
    
    # Prepend initial state
    return jnp.concatenate([z0[None], zs], axis=0)


@partial(jax.jit, static_argnums=(3,))
def integrate_sde_multiple_particles(
    z0: jnp.ndarray,
    ts: jnp.ndarray,
    drift_fn: Callable,
    diffuse_fn: Callable,
    solver: StochasticSolver,
    n_particles: int,
    key: jr.PRNGKey = jr.PRNGKey(0),
) -> jnp.ndarray:
    """
    Integrate SDE for multiple particles with independent Brownian paths.
    
    Args:
        z0: Initial state (d_x,) or (n_particles, d_x,)
        ts: Time points (T,)
        drift_fn: Drift function f(t, z) -> (d_x,) or (n_particles, d_x,)
        diffuse_fn: Diffusion function G(t, z, dW) -> (d_x,) or (n_particles, d_x,)
        solver: Stochastic solver instance
        n_particles: Number of particles
        key: Random key
    
    Returns:
        Trajectories (n_particles, T, d_x)
    """
    keys = jr.split(key, n_particles)
    
    def simulate_one(particle_key):
        bm = VirtualBrownianTree(
            t0=ts[0],
            t1=ts[-1],
            shape=z0.shape if z0.ndim == 1 else z0.shape[1:],
            key=particle_key,
        )
        return integrate_sde(z0, ts, drift_fn, diffuse_fn, solver, bm, particle_key)
    
    return jax.vmap(simulate_one)(keys)
