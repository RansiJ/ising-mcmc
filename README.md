# Ising MCMC

[![CI](https://github.com/RansiJ/ising-mcmc/actions/workflows/ci.yml/badge.svg)](https://github.com/RansiJ/ising-mcmc/actions/workflows/ci.yml)

A compact, reproducible study of the two-dimensional ferromagnetic Ising model
on a periodic square lattice. The central question is whether single-spin
Metropolis-Hastings and heat-bath/Gibbs sampling recover the correct equilibrium
distribution, and how their sampling efficiency changes with temperature.

## Methods and results

Both samplers perform `L**2` random-site updates per sweep and receive an
explicit `numpy.random.Generator`. The physical model, transition kernels, exact
reference, and diagnostics are kept separate.

Correctness is checked against exhaustive enumeration for a 4 x 4 lattice. In
the committed notebook run, both methods reproduce exact mean energy and mean
absolute magnetization at `beta=0.30` and `0.50`, with absolute differences below
approximately 0.022 in the reported intensive observables. The finite-lattice
scan shows the expected increase in magnetic order as beta rises.
Autocorrelation is largest near the theoretical infinite-system reference
`beta_c = 0.5 * log(1 + sqrt(2))`, where both local samplers yield much smaller
effective sample sizes. Runtime and ESS per second are reported separately
because timing depends on the Python implementation and machine.

## Repository structure

```text
src/ising_mcmc/
  model.py          Hamiltonian, observables, and periodic lattice helpers
  samplers.py       Metropolis and Gibbs sweeps
  exact.py          exact enumeration for small validation lattices
  diagnostics.py    autocorrelation time and effective sample size
tests/               focused deterministic tests
scripts/             exact-reference table
notebooks/           main reproducible analysis
```

## Install and run

Python 3.10 or newer is required.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
python scripts/validate_exact_reference.py
jupyter execute notebooks/ising_mcmc.ipynb --inplace --timeout=600
```

The notebook can also be opened and run normally in Jupyter or VS Code.

## Limitations

The simulations use finite lattices and finite chains. The theoretical critical
value is a reference, not a value estimated from the coarse Monte Carlo scan. A
precise critical-point estimate would require a broader finite-size scaling
analysis. Exact enumeration is intentionally restricted to small lattices, and
no cluster algorithms are included.

## License

This project is released under the [MIT License](LICENSE).
