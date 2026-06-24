# Phase-Space Topology and Spectral Flow in Screened Magnetized Plasmas

This repository contains numerical scripts associated with the manuscript:

Published article:

X. Rao, A. Yolbarsop, H. Li, et al.,
Phase-space topology and spectral flow in screened magnetized plasmas,
Phys. Rev. Research 8, 023334 (2026).

Article DOI:
https://doi.org/10.1103/kfhx-hp37

The code is intended to reproduce the main numerical calculations discussed in the paper, including bulk-symbol spectra, Berry--Chern charge calculations, and one-dimensional interface spectral-flow calculations for screened magnetized plasmas.

## Overview

The manuscript studies topological wave phenomena in screened magnetized plasmas using a pseudo-Hermitian formulation and a Weyl-symbol phase-space analysis. The numerical scripts in this repository are provided to improve transparency and reproducibility of the main results.

The repository includes scripts for:

- computing the spectrum of the bulk Weyl symbol;
- calculating Berry--Chern charges associated with isolated degeneracies;
- solving the one-dimensional interface eigenvalue problem;
- generating representative plots of bulk spectra and interface spectral flow.

## Repository structure

```text
screened-magnetized-plasma-topology/
│
├── README.md
├── LICENSE
│
├── band_of_symbol.py
├── Chern_number_calculate.py
├── spectrum_1d_problem.py
│
└── other auxiliary scripts, if included
