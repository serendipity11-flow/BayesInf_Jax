# Third-party notices

Original BayesInf_Jax source and documentation are licensed under the
[MIT License](LICENSE), selected by the repository owner on 2026-10-27.
That grant does not replace third-party copyrights or licenses.

## Included source

**FrEIA 0.2** is vendored under [third_party/FrEIA](third_party/FrEIA) for
implementation comparison. Copyright (c) 2018 Lynton Ardizzone. Its complete
[MIT license](third_party/FrEIA/LICENSE), including its copyright notice and
warranty disclaimer, is retained. The recorded upstream revision is
`421249652c478b3f240f124f69eb5777b6a76bf2` from
[vislearn/FrEIA](https://github.com/vislearn/FrEIA).
It is not required by the JAX model imports.

## Optional upstream repositories

These paths refer to separately maintained Git submodules. Their upstream
licenses and notices remain applicable when the submodules are initialized.

| Component | Recorded revision | Applicable license |
|---|---|---|
| [D4RL](https://github.com/Farama-Foundation/D4RL), `third_party/d4rl_legacy` | `89141a689b0353b0dac3da5cba60da4b1b16254d` | [Apache-2.0 for code](https://github.com/Farama-Foundation/D4RL/blob/master/LICENSE); asset and dataset notices must also be respected |
| [DPPO](https://github.com/irom-princeton/dppo), `third_party/dppo` | `cc7234ad7ff39a8f32de3af903606723a16f0648` | [MIT](https://github.com/irom-princeton/dppo/blob/cc7234ad7ff39a8f32de3af903606723a16f0648/LICENSE), copyright (c) 2024 Intelligent Robot Motion Lab |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), `third_party/mujoco_menagerie` | `71f066ad0be9cd271f7ed58c030243ef157af9f4` | Model-specific licenses. The [ALOHA model](https://github.com/google-deepmind/mujoco_menagerie/blob/main/aloha/LICENSE) used here is BSD-3-Clause, copyright (c) 2023 Trossen Robotics |

D4RL's software license does not grant rights to every dataset. Upstream states
CC-BY-4.0 for its datasets unless otherwise noted; verify the terms of the exact
dataset before acquisition or redistribution. The project MIT license does not
cover external datasets, trained weights, or model assets supplied separately.

## External Python dependencies

These packages are referenced by requirements and installed separately. Their
source, wheels, and native libraries are not bundled in this repository.
The complete direct-pin inventory, installed license-file hashes, and upstream
license links are recorded in [dependency-licenses.json](dependency-licenses.json).

| Packages | Top-level source license |
|---|---|
| JAX, jaxlib, Equinox, Diffrax, Optax, MuJoCo, PyArrow | Apache-2.0 |
| jaxtyping, SALib, pytest | MIT |
| h5py, NumPy, SciPy, scikit-learn, seaborn | BSD-3-Clause |
| Numba | BSD-2-Clause |
| Matplotlib | [Matplotlib license](https://matplotlib.org/3.10.9/project/license.html), based on the PSF license |
| tqdm | [MPL-2.0 and MIT portions](https://github.com/tqdm/tqdm/blob/v4.67.3/LICENCE) |
| Python interpreter | [PSF-2.0 and accompanying notices](https://docs.python.org/3.12/license.html) |

Top-level license names do not summarize every component in upstream binary
distributions. For example, NumPy metadata records additional component licenses,
and SciPy wheels include notices for native libraries and compiler runtimes.
Those bundled notices remain with those distributions. Referencing an unmodified
external dependency does not relicense that dependency under the project MIT
license.

This inventory covers the source distribution and the direct pins in
`requirements.txt`, with Python and jaxlib. `requirements_myenv_312.txt` is a
historical environment freeze, including optional GPU and development packages;
it is not a complete license audit of that environment. Redistributing an
installed environment, container, native library, optional dataset, or model
requires an inventory of that exact artifact and its bundled components.
