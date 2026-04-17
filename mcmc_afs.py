from typing import Iterator

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

RNG = np.random.default_rng()


# Returns the log of the absolute value of the Stirling number of the second kind
def compute_log_stirling(n: int, k: int) -> float:  # AI-generated
	"""
	Computes ln(|S_n^k|) using the recurrence relation in log-space.
	Optimized for memory by only storing the current and previous rows.
	"""
	if k == 0 or k > n:
		return -float("inf")
	if n == 0 and k == 0:
		return 0.0

	# We only need the previous row to calculate the current row
	prev_row = [-float("inf")] * (k + 1)
	prev_row[0] = 0.0  # ln(S(0,0)) = 0

	for row_n in range(1, n + 1):
		curr_row = [-float("inf")] * (k + 1)
		for col_k in range(1, min(row_n, k) + 1):
			# ln_t1 = ln((n-1) * S(n-1, k))
			ln_t1 = np.log(row_n - 1) + prev_row[col_k] if row_n > 1 else -float("inf")
			# ln_t2 = ln(S(n-1, k-1))
			ln_t2 = prev_row[col_k - 1]

			# Log-Sum-Exp Trick to compute ln(exp(ln_t1) + exp(ln_t2))
			if ln_t1 == -float("inf"):
				curr_row[col_k] = ln_t2
			elif ln_t2 == -float("inf"):
				curr_row[col_k] = ln_t1
			else:
				m = max(ln_t1, ln_t2)
				curr_row[col_k] = m + np.log(np.exp(ln_t1 - m) + np.exp(ln_t2 - m))
		prev_row = curr_row

	return float(prev_row[k])


# Returns the log of the total and partial probabilities of the given AFS
def compute_log_probabilities(
	state: npt.NDArray, log_n: float, log_S: float, log_k: float
) -> tuple[float, float]:
	i, alpha_i = np.unique(state, return_counts=True)
	log_alpha_factorials = np.sum(gammaln(alpha_i + 1))
	log_alpha_exponentials = np.sum(alpha_i * np.log(i))
	base = log_n - log_S - log_alpha_exponentials
	log_total_prob = base - log_alpha_factorials
	log_partial_prob = base - log_k

	return float(log_total_prob), float(log_partial_prob)


# Returns the total and partial probabilities of the given AFS
def compute_probabilities(
	state: npt.NDArray, log_n: float, log_S: float, log_k: float
) -> tuple[float, float]:
	log_total, log_partial = compute_log_probabilities(state, log_n, log_S, log_k)
	return float(np.exp(log_total)), float(np.exp(log_partial))


# Returns three numpy arrays:
#   - a 2D numpy array with an AFS in each row
#   - a 1D numpy array with the total probability associated with each AFS in the first array
#   - a 1D numpy array with the partial probability associated with each AFS in the first array
def get_mcmc_allele_frequency_spectra(
	n: int, k: int, num_states: int = 10000, burn_in: int = 1000
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
	assert k > 0
	assert n >= k
	assert num_states > 0
	assert burn_in >= 0

	states = np.zeros((num_states, k), dtype=int)
	total_probabilities = np.zeros(num_states, dtype=float)
	partial_probabilities = np.zeros(num_states, dtype=float)
	log_n = gammaln(n + 1)
	log_S = compute_log_stirling(n, k)
	log_k = gammaln(k + 1)

	current_state = [1] * k
	current_state[0] = n - (k - 1)
	current_state = np.array(current_state, dtype=int)

	for s in range(num_states + burn_in):
		i, j = RNG.integers(k, size=2)
		while i == j:
			i, j = RNG.integers(k, size=2)

		new_state = current_state.copy()

		if new_state[i] > 1:
			new_state[i] -= 1
			new_state[j] += 1

		log_total_prob_current, log_partial_prob_current = compute_log_probabilities(
			current_state, log_n, log_S, log_k
		)
		log_total_prob_new, log_partial_prob_new = compute_log_probabilities(
			new_state, log_n, log_S, log_k
		)
		log_u = np.log(RNG.random())

		accept_transition = log_u < log_partial_prob_new - log_partial_prob_current
		if accept_transition:
			current_state = new_state
			log_total_prob_current = log_total_prob_new
			log_partial_prob_current = log_partial_prob_new

		if s >= burn_in:
			states[s - burn_in] = np.sort(current_state)
			total_probabilities[s - burn_in] = np.exp(log_total_prob_current)
			partial_probabilities[s - burn_in] = np.exp(log_partial_prob_current)

	return states, total_probabilities, partial_probabilities


# Returns the homozygosity of the given AFS
def compute_homozygosity(state: npt.NDArray) -> float:
	return float((state**2).sum() / (state.sum() ** 2))


# Returns the homozygosities of the given AFSs
def compute_homozygosities(states: npt.NDArray) -> npt.NDArray:
	return (states**2).sum(axis=1) / (states.sum(axis=1) ** 2)


# Returns the expected homozygosity for a given sample size and number of allelic classes
# If mcmc is True, the expected homozygosity is estimated using a MCMC algorithm
#   The return value is a tuple with the mean and standard deviation of the estimate
# If mcmc is False, the expected homozygosity is computed using brute force
#   The return value is a tuple with the actual expected homozygosity and 0.0
def compute_expected_homozygosity(
	n: int,
	k: int,
	mcmc: bool = True,
	num_states: int = 100000,
	num_replicates: int = 10,
) -> tuple[float, float]:
	if mcmc:
		rep_results = np.zeros(num_replicates)

		for i in range(num_replicates):
			states, _, _ = get_mcmc_allele_frequency_spectra(n, k, num_states)
			rep_results[i] = compute_homozygosities(states).mean()

		return float(rep_results.mean()), float(rep_results.std())

	else:
		log_n = gammaln(n + 1)
		log_S = compute_log_stirling(n, k)
		log_k = gammaln(k + 1)

		homozygosities = list()
		total_probabilities = list()

		for state in _partitions_n_k(n, k):
			state = np.array(state)
			total_prob, _ = compute_probabilities(state, log_n, log_S, log_k)
			homozygosities.append(compute_homozygosity(state))
			total_probabilities.append(total_prob)

		homozygosities = np.array(homozygosities)
		total_probabilities = np.array(total_probabilities)
		return float((homozygosities * total_probabilities).sum()), 0.0


# Returns the exact neutrality test (Slatkin 1994) probability of the given AFS
# If mcmc is True, the neutrality test is estimated using a MCMC algorithm
#   The return value is a tuple with the mean and standard deviation of the estimate
# If mcmc is False, the neutrality test is computed using brute force
#   The return value is a tuple with the actual probability and 0.0
def compute_exact_neutrality_test(
	c0: npt.NDArray | list[int],
	mcmc: bool = True,
	num_states: int = 100000,
	num_replicates: int = 10,
) -> tuple[float, float]:
	c0 = np.array(c0)
	n = c0.sum()
	k = c0.size

	log_n = gammaln(n + 1)
	log_S = compute_log_stirling(n, k)
	log_k = gammaln(k + 1)
	_, partial_prob_c0 = compute_probabilities(c0, log_n, log_S, log_k)

	if mcmc:
		rep_results = np.zeros(num_replicates)

		for i in range(num_replicates):
			_, total_probabilities, partial_probabilities = (
				get_mcmc_allele_frequency_spectra(n, k, num_states)
			)
			rep_results[i] = (partial_probabilities <= partial_prob_c0).mean()

		return float(rep_results.mean()), float(rep_results.std())

	else:
		total_probabilities = list()
		partial_probabilities = list()

		for state in _partitions_n_k(n, k):
			state = np.array(state)
			total_prob, partial_prob = compute_probabilities(state, log_n, log_S, log_k)
			total_probabilities.append(total_prob)
			partial_probabilities.append(partial_prob)

		total_probabilities = np.array(total_probabilities)
		partial_probabilities = np.array(partial_probabilities)
		return float(
			total_probabilities[partial_probabilities <= partial_prob_c0].sum()
		), 0.0


def _partitions_n_k(
	n: int, k: int, min_val: int = 1
) -> Iterator[tuple[int, ...]]:  # AI-generated
	"""
	Generator that yields all partitions of n into exactly k parts.
	Each part is >= min_val to ensure uniqueness and order.
	"""
	# Base Case: If we need 1 part, it must be the remaining n
	if k == 1:
		if n >= min_val:
			yield (n,)
		return

	# Recursive Step:
	# The smallest possible value for the current part is 'min_val'.
	# The largest possible value is n // k (to leave enough for remaining parts).
	for i in range(min_val, n // k + 1):
		for p in _partitions_n_k(n - i, k - 1, i):
			yield (i,) + p


if __name__ == "__main__":
	from time import time

	for c0 in (
		[9, 2, 1, 1, 1, 1, 1],
		[42, 14, 4, 3, 3, 2, 1, 1, 1, 1],
		[52, 9, 8, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
	):
		print(c0)
		print(f"n = {sum(c0)}\t k = {len(c0)}")

		print("* Expected homozygosity *")

		t0 = time()
		exp_homo = compute_expected_homozygosity(sum(c0), len(c0), mcmc=False)
		exp_homo_mcmc = compute_expected_homozygosity(
			sum(c0), len(c0), num_states=1000000, num_replicates=30
		)

		print(f"\treal: {exp_homo}")
		print(f"\tmcmc: {exp_homo_mcmc}")

		total_time = time() - t0
		hours = int(total_time // 3600)
		minutes = int((total_time % 3600) // 60)
		seconds = total_time % 60
		print(f"Computed in {hours:02d}h{minutes:02d}m{seconds:05.2f}s.")

		print()

		print("* Neutrality test *")

		t0 = time()
		neu_test = compute_exact_neutrality_test(c0, mcmc=False)
		neu_test_mcmc = compute_exact_neutrality_test(
			c0, num_states=1000000, num_replicates=30
		)

		print(f"\treal: {neu_test}")
		print(f"\tmcmc: {neu_test_mcmc}")

		total_time = time() - t0
		hours = int(total_time // 3600)
		minutes = int((total_time % 3600) // 60)
		seconds = total_time % 60
		print(f"Computed in {hours:02d}h{minutes:02d}m{seconds:05.2f}s.")

		print()

	n, k = 100, 15
	reps = 5

	print(f"n = {n}, k = {k}")
	print()

	print("* Expected homozygosity *")

	t0 = time()
	exp_homo = compute_expected_homozygosity(n, k, mcmc=False)
	exp_homo_mcmc = compute_expected_homozygosity(
		n, k, num_states=1000000, num_replicates=30
	)

	print(f"\treal: {exp_homo}")
	print(f"\tmcmc: {exp_homo_mcmc}")

	total_time = time() - t0
	hours = int(total_time // 3600)
	minutes = int((total_time % 3600) // 60)
	seconds = total_time % 60
	print(f"Computed in {hours:02d}h{minutes:02d}m{seconds:05.2f}s.")

	print()

	print("* Neutrality test *")
	for _ in range(reps):
		c0 = get_mcmc_allele_frequency_spectra(n, k, num_states=1, burn_in=2000)[0][0]

		print(f"\t{c0}")

		t0 = time()
		neu_test = compute_exact_neutrality_test(c0, mcmc=False)
		neu_test_mcmc = compute_exact_neutrality_test(
			c0, num_states=1000000, num_replicates=30
		)

		print(f"\treal: {neu_test}")
		print(f"\tmcmc: {neu_test_mcmc}")

		total_time = time() - t0
		hours = int(total_time // 3600)
		minutes = int((total_time % 3600) // 60)
		seconds = total_time % 60
		print(f"Computed in {hours:02d}h{minutes:02d}m{seconds:05.2f}s.")

		print()
