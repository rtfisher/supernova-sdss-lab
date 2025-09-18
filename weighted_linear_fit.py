#!/usr/bin/env python3
# Weighted linear regression with heteroscedastic errors.
#
# Logic:
# 1) Generate (or load) data (x_i, y_i) with known per-point y uncertainties sigma_i.
# 2) Perform a Weighted Least Squares (WLS) fit for y = m x + b using weights w_i = 1/sigma_i^2.
#    Solve (X^T W X) beta = X^T W y for beta = [m, b].
# 3) Compute parameter covariance. If sigmas are accurate, cov = (X^T W X)^(-1).
#    If there is extra scatter, scale by reduced chi^2: cov *= chi2/(N - p).
# 4) Print best-fit slope and intercept with 1σ error bars propagated from the covariance.
# 5) Plot the data with error bars and overlay the best-fit line.
#
# Replace the synthetic data block with your own data as needed.
# Requires: numpy, matplotlib.

import numpy as np
import matplotlib.pyplot as plt

def weighted_linear_fit(x, y, y_err):
    """Return m, b, sigma_m, sigma_b, chi2, red_chi2 for y = m x + b."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray(y_err, dtype=float)

    w = 1.0 / (y_err**2)
    X = np.vstack([x, np.ones_like(x)]).T  # design matrix (N,2)
    W = np.diag(w)

    XTWX = X.T @ W @ X
    XTWX_inv = np.linalg.inv(XTWX)
    beta = XTWX_inv @ (X.T @ W @ y)

    m_fit, b_fit = beta
    resid = y - (m_fit * x + b_fit)
    chi2 = np.sum((resid / y_err)**2)
    dof = len(x) - 2
    red_chi2 = chi2 / dof

    # Parameter covariance (scaled by reduced chi^2)
    cov = XTWX_inv * red_chi2
    sigma_m, sigma_b = np.sqrt(np.diag(cov))

    return m_fit, b_fit, sigma_m, sigma_b, chi2, red_chi2

def main():
    # ---- Synthetic example (replace with your data) ----
    rng = np.random.default_rng(8)
    true_m, true_b = 2.3, -0.7
    x = np.linspace(0, 10, 25)
    y_err = 0.5 + 0.15 * x  # heteroscedastic errors
    y_true = true_m * x + true_b
    y = y_true + rng.normal(0, y_err)

    # ---- Fit ----
    m, b, sm, sb, chi2, rchi2 = weighted_linear_fit(x, y, y_err)

    # ---- Print coefficients WITH error bars ----
    print(f"Slope m = {m:.4f} ± {sm:.4f}")
    print(f"Intercept b = {b:.4f} ± {sb:.4f}")
    print(f"chi^2 = {chi2:.2f}, dof = {len(x)-2}, reduced chi^2 = {rchi2:.3f}")

    # ---- Plot ----
    plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=3, label='Data')
    x_plot = np.linspace(x.min(), x.max(), 300)
    plt.plot(x_plot, m * x_plot + b, label='Weighted fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Weighted Linear Fit with Heteroscedastic Errors')
    plt.legend()
    plt.tight_layout()
    plt.savefig('weighted_fit.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    main()

