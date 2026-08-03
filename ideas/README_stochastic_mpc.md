# Why is the stochastic MPC evaluation so slow?

## 1. Computational scaling

For every MPC command, CEM performs

N_iter × N_candidates × N_particles × Horizon

With the current settings:

- CEM iterations: 5
- Candidates: 256
- Particles: 12
- Horizon: 25

This gives:

5 × 256 × 12 × 25 = 384000

One control command therefore requires approximately **384,000 model transitions**.

---

## 2. Total evaluation cost

The OOD benchmark runs:

- 5 scenarios
- 2 controllers
- 6 episodes
- 120 control steps

Total MPC calls:

5 × 2 × 6 × 120 = 7200

Total transitions:

7200 × 384000 ≈ 2.8 billion transitions.

---

## 3. LatentSSMFlow is expensive

Each transition contains:

- GRU updates
- latent-state propagation
- normalizing-flow coupling blocks
- stochastic sampling

The cost scales as

O(iterations × candidates × particles × horizon × model_cost)

where model_cost is much larger than for the nominal model.

---

## 4. Only the cost function is JIT compiled

The cost kernel is JIT compiled, but the outer CEM loop still runs in Python.

Consequences:

- repeated device dispatches
- repeated sorting
- extra synchronization overhead

The entire CEM optimization should ideally be moved into a single JAX function using `lax.scan`.

---

## 5. Python loops increase compilation time

The nominal rollout and particle rollout still use Python loops.

They should be replaced by `jax.lax.scan`, which reduces graph size and compilation time.

---

## 6. Synchronization overhead

The code calls:

```python
action.block_until_ready()
```

after every command.

This gives correct timing but forces synchronization after each MPC step.

---

## 7. Extra evaluation before closed-loop testing

The script additionally runs:

- open-loop evaluation
- throughput benchmarks
- plotting
- serialization

before closed-loop evaluation even starts.

---

# Recommended development settings

```text
horizon = 15
candidates = 64
particles = 4
cem_iterations = 3
closed_loop_steps = 60
episodes = 2
```

The new workload becomes:

```text
3 × 64 × 4 × 15 = 11520
```

which is about 33× cheaper.

---

# Recommended improvements

1. JIT the entire CEM loop.
2. Replace Python loops with `jax.lax.scan`.
3. Use partial sorting instead of `argsort`.
4. Reduce particles and candidates during debugging.
5. Add flags such as:

```text
--skip-open-loop
--skip-timing
--closed-loop-only
```

The main bottleneck is the multiplication:

```text
candidates × particles × horizon × iterations × steps × episodes × scenarios
```

combined with the computational cost of LatentSSMFlow.
