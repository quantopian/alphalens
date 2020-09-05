import numpy as np
import scipy.integrate as integrate


def _compute_vhat(ret1, ret2):
    nu = [np.mean(ret1), np.mean(ret2), np.mean(ret1 ** 2), np.mean(ret2 ** 2)]
    nux = np.vstack([ret1 - nu[0], ret2 - nu[1], ret1 ** 2 - nu[2], ret2 ** 2 - nu[3]]).T
    return nux


def _ar2(y, nlag, const=1):
    # ols estimates for the AR(k) model
    results = {}
    n = y.shape[0]
    results['y'] = y
    results['negs'] = 1
    cols = []
    if const == 1:
        cols.append(np.ones(n))
    for i in range(nlag):
        cols.append(np.roll(y, i + 1))

    x = np.vstack(cols).T
    x = x[nlag:, :]
    y = y[nlag:]
    n_adj = len(y)

    b0 = np.linalg.lstsq(x.T.dot(x), x.T.dot(y), rcond=None)[0]
    p = nlag + const
    sige = ((y - x.dot(b0)).T.dot(y - x.dot(b0))) / (n - p + 1)

    results['meth'] = 'ar'
    results['beta'] = b0
    results['sige'] = sige
    results['yhat'] = x.dot(b0)
    results['nobs'] = n
    results['nadj'] = n_adj
    results['nvar'] = nlag * const
    results['x'] = x
    return results


def _compute_alpha(v_hat):
    t = v_hat.shape[0]
    p = v_hat.shape[1]
    numerator = 0
    denominator = 0
    for i in range(p):
        results = _ar2(v_hat[:, i], 1)
        rho_hat = results['beta'][1]
        sig_hat = np.sqrt(results['sige'])
        numerator = numerator + 4 * rho_hat ** 2 * sig_hat ** 4 / (1 - rho_hat) ** 8
        denominator = denominator + sig_hat ** 4 / (1 - rho_hat) ** 4

    alpha_hat = numerator / denominator
    return alpha_hat


def _gamma_hat(v_hat, j):
    t = v_hat.shape[0]
    p = v_hat.shape[1]
    gamma_hat = np.zeros((p, p))
    if j >= t:
        raise Exception('j must be smaller than the row dimension!')
    else:
        for i in range(j, t):
            gamma_hat = gamma_hat + v_hat[i, :].reshape(-1, 1).dot(v_hat[i - j, :].reshape(1, -1))
        gamma_hat = gamma_hat / t
        return gamma_hat


def _kernel_type(x, type):
    if type == 'G':
        if x < 0.5:
            wt = 1 - 6 * x ** 2 + 6 * x ** 3
        elif x < 1:
            wt = 2 * (1 - x) ** 3
        else:
            wt = 0
        return wt
    elif type == 'QS':
        term = 6 * np.pi * x / 5
        wt = 25 * (np.sin(term) / term - np.cos(term)) / (12 * np.pi ** 2 * x ** 2)
        return wt
    else:
        raise Exception('wrong type')


def _compute_psi(v_hat):
    t = v_hat.shape[0]
    alpha_hat = _compute_alpha(v_hat)
    ss_star = 2.6614 * (alpha_hat * t) ** 0.2
    psi_hat = _gamma_hat(v_hat, 0)
    j = 1
    while j < ss_star:
        gamma = _gamma_hat(v_hat, j)
        psi_hat = psi_hat + _kernel_type(j / ss_star, "G") * (gamma + gamma.T)
        j += 1
    psi_hat = (t / (t - 4)) * psi_hat
    return psi_hat


def _compute_se(ret1, ret2):
    t = len(ret1)
    mu1_hat = np.mean(ret1)
    mu2_hat = np.mean(ret2)
    gamma1_hat = np.mean(ret1 ** 2)
    gamma2_hat = np.mean(ret2 ** 2)
    gradient = np.zeros((4, 1))
    gradient[0] = gamma1_hat / (gamma1_hat - mu1_hat ** 2) ** 1.5
    gradient[1] = -gamma2_hat / (gamma2_hat - mu2_hat ** 2) ** 1.5
    gradient[2] = -0.5 * mu1_hat / (gamma1_hat - mu1_hat ** 2) ** 1.5
    gradient[3] = 0.5 * mu2_hat / (gamma2_hat - mu2_hat ** 2) ** 1.5
    v_hat = _compute_vhat(ret1, ret2)
    psi_hat = _compute_psi(v_hat)
    se = np.sqrt(gradient.T.dot(psi_hat).dot(gradient) / t)
    return se


def sharpe_hac(ret1, ret2):
    """
    This method performs a sharpe test on two return sequences,
    returns:
    diff: the difference in sharpe (sharpe1 - sharpe2)
    pval: P-value under the assumption that H0 holds， H0: there is no difference between these two sharpes


    For mathematical principles, please refer to the paper: http://www.ledoit.net/jef_2008pdf.pdf
    """
    mu1_hat = np.mean(ret1)
    mu2_hat = np.mean(ret2)
    sig1_hat = np.std(ret1)
    sig2_hat = np.std(ret2)
    sr1_hat = mu1_hat / sig1_hat
    sr2_hat = mu2_hat / sig2_hat
    diff = sr1_hat - sr2_hat
    se = _compute_se(ret1, ret2)[0, 0]
    func = lambda x: (1 / np.sqrt(np.pi * 2)) * np.exp(-0.5 * x ** 2)
    pval, err = integrate.quad(func, -1000, -abs(diff) / se)
    pval = 2 * pval
    return diff, pval
