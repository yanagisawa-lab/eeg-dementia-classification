#!/usr/bin/env python3
# Time-stamp: "2021-09-25 15:39:38 (ylab)"

import numpy as np
import torch
import mngs


def fdr_correction(pvals, alpha=0.05, method="indep"):
    # https://github.com/mne-tools/mne-python/blob/main/mne/stats/multi_comp.py
    """P-value correction with False Discovery Rate (FDR).

    Correction for multiple comparison using FDR :footcite:`GenoveseEtAl2002`.

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like
        Set of p-values of the individual tests.
    alpha : float
        Error rate.
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not.
    pval_corrected : array
    alpha : float
        Error rate.
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not.
    pval_corrected : array
    alpha : float
        Error rate.
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not.
    pval_corrected : array
        P-values adjusted for multiple hypothesis testing to limit FDR.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not.
    pval_corrected : array
        P-values adjusted for multiple hypothesis testing to limit FDR.

    References
    ----------
    .. footbibliography::
    """
    pvals = np.asarray(pvals)
    shape_init = pvals.shape
    pvals = pvals.ravel()

    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()

    if method in ["i", "indep", "p", "poscorr"]:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ["n", "negcorr"]:
        cm = np.sum(1.0 / np.arange(1, len(pvals_sorted) + 1))
        ecdffactor = _ecdf(pvals_sorted) / cm
    else:
        raise ValueError("Method should be 'indep' and 'negcorr'")

    reject = pvals_sorted < (ecdffactor * alpha)
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    pvals_corrected = pvals_corrected[sortrevind].reshape(shape_init)
    reject = reject[sortrevind].reshape(shape_init)
    return reject, pvals_corrected


def _ecdf(x):
    # https://github.com/mne-tools/mne-python/blob/main/mne/stats/multi_comp.py
    """No frills empirical cdf used in fdrcorrection."""
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


def _ecdf_torch(x):
    """No frills empirical cdf used in fdrcorrection."""
    nobs = len(x)
    return torch.arange(1, nobs + 1) / float(nobs)


def fdr_correction_torch(pvals, alpha=0.05, method="indep"):
    """P-value correction with False Discovery Rate (FDR).

    Correction for multiple comparison using FDR :footcite:`GenoveseEtAl2002`.

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like
        Set of p-values of the individual tests.
    alpha : float
        Error rate.
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not.
    pval_corrected : array
    alpha : float
        Error rate.
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not.
    pval_corrected : array
    alpha : float
        Error rate.
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not.
    pval_corrected : array
        P-values adjusted for multiple hypothesis testing to limit FDR.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not.
    pval_corrected : array
        P-values adjusted for multiple hypothesis testing to limit FDR.

    References
    ----------
    .. footbibliography::
    """

    shape_init = pvals.shape
    pvals = pvals.ravel()

    pvals_sortind = torch.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()

    if method in ["i", "indep", "p", "poscorr"]:
        ecdffactor = _ecdf_torch(pvals_sorted)
    elif method in ["n", "negcorr"]:
        cm = torch.sum(1.0 / torch.arange(1, len(pvals_sorted) + 1))
        ecdffactor = _ecdf_torch(pvals_sorted) / cm
    else:
        raise ValueError("Method should be 'indep' and 'negcorr'")

    ecdffactor = ecdffactor.type_as(pvals_sorted)

    reject = pvals_sorted < (ecdffactor * alpha)

    if reject.any():
        rejectmax = max(torch.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected_raw_flipped = pvals_corrected_raw.flip(0)
    pvals_cum_min = torch.zeros_like(pvals_corrected_raw_flipped)
    for ii in range(len(pvals_cum_min)):
        pvals_cum_min[ii] = (pvals_corrected_raw_flipped[0 : int(ii) + 1]).min()
    pvals_corrected = pvals_cum_min.flip(0)

    pvals_corrected[pvals_corrected > 1.0] = 1.0
    pvals_corrected = pvals_corrected[sortrevind].reshape(shape_init)
    reject = reject[sortrevind].reshape(shape_init)
    return reject, pvals_corrected


if __name__ == "__main__":
    pvals = [0.02, 0.03, 0.05]
    pvals_torch = torch.tensor(np.array([0.02, 0.03, 0.05]))

    reject, pvals_corrected = fdr_correction(pvals, alpha=0.05, method="indep")

    reject_torch, pvals_corrected_torch = fdr_correction_torch(
        pvals, alpha=0.05, method="indep"
    )

    arr = pvals_corrected.astype(float)
    tor = pvals_corrected_torch.numpy().astype(float)
    print(mngs.general.isclose(arr, tor))
