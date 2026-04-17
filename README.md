# MCMC AFS

This script allows you to use an MCMC algorithm to estimate Alle Frequency Spectra measurements, such as the expected homozygosity for a given sample size and number of alleleic classes and the Slatkin's (1994) exact neutrality test.

## Key features

- Generate a sequence of Alle Frequency Spectra using an MCMC algorithm.
- Compute the expected homozygosity using brute force or estimate ir using an MCMC algorithm.
- Compute the exact neutrality test using brute force or estimate ir using an MCMC algorithm.

## Requirements

- Python 3.10.2 or newer
- NumPy
- SciPy

## Installation

Copy the `mcmc_afs.py` script into your working directory.

## Usage

Run the `mcmc_afs.py` script directly to see a performance test.

Import the script or the functions you intend to use into your own python script:
```python
import mcmc_afs
from mcmc_afs import compute_exact_neutrality_test
```

## What I want to add...

1. a `MCMCAFS` class, so that the user can run the MCMC algorithm only once and compute any measurement in a more straightforward way.
2. more options of measurements.
3. a seed setter, for reproducibility.
4. a way for the user to save the `MCMCAFS` object.
5. parallelization.
