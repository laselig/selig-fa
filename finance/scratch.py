import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")
import numpy as np 
import matplotlib.pyplot as plt
def sigmoid(x, alpha, beta):
    """Standard sigmoid function, translated by beta and scaled by alpha.
    With positive alpha, approaches 0 as x -> -inf and 1 as x -> inf.
    Equals 1/2 at x = beta and smoothly interpolates between these values.

    Parameters
    ----------
    x : float
        Value to fix between 0 - 1
    alpha : float
        Steepness of the function. Slope is alpha/4 at x = beta.
        Valid for all real values.
        Note: negative values will switch the limits.
    beta : float
        Translates function along the X-axis. Curve is centered at x = beta.
        Valid for all real values.

    Returns
    -------
    float
        Returns value constrained between 0 and 1.
    """

    arg = alpha * (x - beta)
    # if abs(arg) of exp is >100 return 0 or 1,
    # prevents over/underflow
    if arg > 100:
        return 1.0
    elif -arg > 100:
        return 0.0
    else:
        # Actual sigmoid function for reasonable values
        return 1.0 / (1.0 + np.exp(-arg))

def asymmetric_booster(x, alpha, beta, gamma):

    """Asymmetric booster function, translated by beta, scaled by alpha, and
    pushed along Y-axis by gamma.
    With positive alpha, approaches 0 as x -> -inf and 100 as x -> inf.
    Smoothly interpolates between these values.
    Note: Equals gamma at x = beta.

    Parameters
    ----------
    x : float OR np.array(float)
        Input stress (or array of stress values) to fix between 0 - 100
    alpha : float
        Steepness of the function. Slope is 25*alpha/2 at x = beta.
        Valid for all real values, but > 0 for typical behavior.
    beta : float
        Translates function along the X-axis. Curve is centered at x = beta
        Valid for all real values.
    gamma : float
        'Pushes' curve along Y-axis,
        pushed up for gamma>50,and down for gamma<50.
        Valid for range (25,75) excluding endpoints.

    Returns
    -------
    float
        Returns stress value, constrained between 0 and 100.
    """
    if hasattr(x, "__iter__"):
        return np.array([asymmetric_booster(_x, alpha, beta, gamma) for _x in x])

    if (gamma < 25) or (gamma > 75):
        raise Exception(f"gamma={gamma} is outside range (25,75)")

    tol = 1e-10

    if gamma < 25.0 + tol:
        log.debug(f"gamma={gamma} close to 25. Replacing with 25 + {tol}")
        gamma = 25.0 + tol
    elif 75.0 - gamma < tol:
        log.debug(f"gamma={gamma} close to 75. Replacing with 75 - {tol}")
        gamma = 75.0 - tol

    # Rescale gamma to ensure f(stress=beta) = gamma
    gamma = 2 * (gamma - 25)

    # k regulates cutover from x < beta to x > beta behavior
    # For gamma further from 50, cutover should occur more quickly,
    # pushing the curve along the Y-axis
    d = np.abs(gamma - 50) + 50
    k = 1 + np.log(d / (100 - d))

    # scale alpha such that slope at x = beta is unaffected by gamma
    alpha = alpha / ((k + 1))

    # Cutover functions
    A = sigmoid(x, alpha * k, beta)  # 1 at x >> beta, 0 at x << beta
    B = sigmoid(x, -alpha * k, beta)  # 1 at x << beta. 0 at x >> beta

    # x >> beta behavior
    f_largex = (100 - gamma) * sigmoid(x, alpha, beta) + gamma

    # x << beta behavior
    f_smallx = gamma * sigmoid(x, alpha, beta)

    score = A * f_largex + B * f_smallx

    return score
# df = pd.read_parquet("/Users/lselig/Desktop/tmp_dir/features.parquet")
# alpha = 14.82
# beta = 0.402
# gamma = 38.3
# ws = df.weighted_sum.values
# ws = ws[np.where(~np.isnan(ws))]
# scores = asymmetric_booster(ws, alpha, beta, gamma)
# plt.plot(scores)
# plt.show()

eda_df = pd.read_parquet("/Users/lselig/Desktop/tmp_dir/STAGING_PRS_harlie_staging/prs_window_eda_1659390693.parquet")
features_df = pd.read_parquet("/Users/lselig/Desktop/tmp_dir/STAGING_PRS_harlie_staging/features_1659390693.parquet")

fig, axs = plt.subplots(3, 1, figsize = (15, 9), sharex = True)
axs[0].plot(eda_df.dt_tmp, eda_df.conductance)
axs[0].set_ylabel("EDA (µS)")
axs[1].plot(eda_df.dt_tmp, eda_df.drive_voltage / 1e12)
axs[1].set_ylabel("Drive Voltge (V)")
axs[2].plot(pd.to_datetime(features_df.ts, unit = "s"), features_df.acr)
axs[2].set_ylabel("ACR capped at 10")
axs[2].set_ylim(0, 10)
axs[2].set_title(f"Median ACR: {np.nanmedian(features_df.acr):.2f}")
# axs[3].plot(pd.to_datetime(features_df.ts, unit = "s"), features_df.sqi, drawstyle = "steps-post")
# axs[3].set_ylabel("SQI")
fig.suptitle("Harlie personalization window")
plt.show()