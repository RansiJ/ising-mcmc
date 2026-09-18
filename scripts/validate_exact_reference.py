"""Generate deterministic exact-equilibrium reference values for small lattices."""

from __future__ import annotations

from ising_mcmc.exact import exact_statistics


def main() -> None:
    cases = [
        (2, 0.20),
        (2, 0.44),
        (3, 0.20),
        (3, 0.44),
    ]

    header = (
        "L  beta      <E>/N       <|M|>/N       C/N          chi/N"
    )
    print(header)
    print("-" * len(header))

    for L, beta in cases:
        stats = exact_statistics(L=L, beta=beta, J=1.0, h=0.0)
        print(
            f"{L:<2d} {beta:>5.2f}  "
            f"{stats.energy_density:>11.6f}  "
            f"{stats.abs_magnetization_density:>11.6f}  "
            f"{stats.heat_capacity_per_spin:>11.6f}  "
            f"{stats.susceptibility_per_spin:>11.6f}"
        )


if __name__ == "__main__":
    main()
