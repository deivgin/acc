# Wind Observer — Numerical Issues

Two known numerical weaknesses in the current Kalman filter implementation
(`src/acc/aero/wind_observer.py`) that should be addressed before processing
long or noisy real-world flight logs.

---

## 1. Covariance update loses positive-definiteness

### What happens

The covariance update uses the **basic (standard) form**:

```
P ← (I − K H) P
```

Over many iterations this expression is numerically unstable.  Each update
introduces small floating-point round-off errors that accumulate and can:

1. Break the symmetry of P (P ≠ P^T up to machine precision).
2. Push small eigenvalues negative, making P **indefinite**.

Once P is indefinite the Kalman gain computation `K = P H / S` produces
incorrect (and sometimes enormous) gains.  The state estimate then diverges
or oscillates wildly.

### Why it happens

The issue is algebraic cancellation.  Expanding the update:

```
P_new = P − K H P
```

The term `K H P` is meant to exactly cancel the portion of P that has been
"explained" by the measurement.  In exact arithmetic this works.  In
finite-precision arithmetic, `K H P` and `P` are of similar magnitude, so
their difference loses significant digits — the same phenomenon as computing
`1.0000001 − 1.0000000` in float64.

The effect is worst when:

- The Kalman gain is large (high-confidence measurement, small S).
- P has eigenvalues spanning many orders of magnitude (common when some
  states are well-observed and others are not — exactly the case here with
  v_wind_D being weakly observable in level flight).
- The filter runs for thousands of steps (a 10-minute flight at 50 Hz is
  30 000 steps).

### Fix: Joseph (stabilised) form

Replace the basic update with the **Joseph form**:

```
P ← (I − K H) P (I − K H)^T + K R K^T
```

This is mathematically equivalent but numerically superior because:

- The first term `(I − K H) P (I − K H)^T` is a congruence transformation,
  which **guarantees** the result is symmetric and positive-semi-definite
  whenever P is.  (For any matrix M and PSD matrix P, `M P M^T` is PSD.)
- The second term `K R K^T` adds a small positive-definite correction that
  compensates for the finite-precision loss in the first term.

In scalar-measurement form (as in this observer), the implementation is:

```python
IKH = identity - np.outer(kalman_gain, H)
covariance = IKH @ covariance @ IKH.T + R_meas * np.outer(kalman_gain, kalman_gain)
```

The cost is one extra outer product and matrix multiply per step — negligible
for a 4×4 system.

### References

- Bucy, R. S. and Joseph, P. D., *Filtering for Stochastic Processes with
  Applications to Guidance*, 1968.
- Grewal, M. S. and Andrews, A. P., *Kalman Filtering: Theory and Practice
  Using MATLAB*, 4th ed., Chapter 6.

---

## 2. No guard against degenerate innovation variance

### What happens

The innovation (residual) variance is computed as:

```python
innovation_var = H @ covariance @ H + R_meas
```

The Kalman gain is then:

```python
kalman_gain = (covariance @ H) / innovation_var
```

If `innovation_var` is very small or (due to the covariance issue above)
negative, the division produces an enormous or NaN gain.  A single corrupted
gain step can permanently destabilise the filter.

### When it happens

- **Covariance collapse**: after many low-noise measurements, P shrinks until
  `H P H^T` underflows to zero.  Then `innovation_var ≈ R_meas`, which is
  fine — but if R_meas is also set very small (e.g. during tuning), S can
  become tiny.
- **Indefinite P** (issue 1 above): `H P H^T` can be negative, and if
  `|H P H^T| > R_meas` then `innovation_var < 0`, producing a negative gain
  that inverts the correction direction.
- **Near-singular geometry**: when pitch ≈ ±90° (vertical dive/climb),
  `cos θ ≈ 0` so H ≈ [0, 0, ∓1, pitot].  This doesn't make S zero by
  itself, but combined with a collapsed P it can.

### Fix: floor clamp on innovation variance

Add a lower bound before computing the gain:

```python
innovation_var = max(H @ covariance @ H + R_meas, 1e-12)
```

This ensures the division is always well-defined and the gain magnitude is
bounded.  The value `1e-12` is small enough to never interfere with normal
operation (typical S values are O(1) to O(10)) but large enough to prevent
float64 overflow in `1/S`.

An alternative is to skip the measurement update entirely when `innovation_var`
is below a threshold, effectively treating that time step as a predict-only
step.  This is more conservative but avoids any risk of a bad update:

```python
if innovation_var < 1e-9:
    state_history[k] = state_estimate
    continue
```

Either approach prevents silent corruption of the state estimate.
