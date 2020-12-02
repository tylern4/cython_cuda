from scipy import stats
import boost_histogram as bh
import matplotlib.pyplot as plt
import time
import wrapper
import pandas as pd
import numpy as np
np.warnings.filterwarnings('ignore')


@np.vectorize
def slow_w(e_p, e_theta, e_phi):
    MP = 0.93827208816
    E0 = 4.81726
    ME = 0.00051099895

    p_targ_px = 0.0
    p_targ_py = 0.0
    p_targ_pz = 0.0
    p_targ_E = MP

    e_beam_px = 0.0
    e_beam_py = 0.0
    e_beam_pz = np.sqrt(E0**2-ME**2)
    e_beam_E = E0
    e_prime_px = e_p*np.sin(e_theta)*np.cos(e_phi)
    e_prime_py = e_p*np.sin(e_theta)*np.sin(e_phi)
    e_prime_pz = e_p*np.cos(e_theta)
    e_prime_E = np.sqrt(e_prime_px**2 + e_prime_py**2 + e_prime_pz**2 - ME**2)

    temp_px = e_beam_px - e_prime_px + p_targ_px
    temp_py = e_beam_py - e_prime_py + p_targ_py
    temp_pz = e_beam_pz - e_prime_pz + p_targ_pz
    temp_E = e_beam_E - e_prime_E + p_targ_E

    temp2 = temp_px**2+temp_py**2+temp_pz**2-temp_E**2
    temp3 = np.sqrt(-temp2)

    return temp3


if __name__ == "__main__":
    # wrapper.cuda_properties()

    df = pd.read_csv("/mnt/ssd/kincorr/momCorr.csv")

    print("size ", df.e_p.to_numpy().size)

    # start = time.time()
    # w = slow_w(df.e_p.to_numpy(), np.deg2rad(
    #     df.e_theta), np.deg2rad(df.e_phi))
    # total = time.time() - start
    # print("Slow W: ", total, "sec", (df.e_p.to_numpy().size/total)/1E6, "MHz")

    start = time.time()
    w = wrapper.cython_w(df.e_p.to_numpy(),
                         np.deg2rad(df.e_theta), np.deg2rad(df.e_phi))
    total = time.time() - start
    print("cython W: ", total, "sec", (df.e_p.to_numpy().size/total)/1E6, "MHz")

    start = time.time()
    w = wrapper.cuda_w(4.81726, df.e_p.to_numpy(),
                       np.deg2rad(df.e_theta), np.deg2rad(df.e_phi))
    total = time.time() - start
    print("cuda W: ", total, "sec", (df.e_p.to_numpy().size/total)/1E6, "MHz")

    start = time.time()
    q2 = wrapper.cuda_q2(4.81726, df.e_p.to_numpy(),
                         np.deg2rad(df.e_theta), np.deg2rad(df.e_phi))
    total = time.time() - start
    print("cuda Q2: ", total, "sec", (df.e_p.to_numpy().size/total)/1E6, "MHz")

    y, x = bh.numpy.histogram(w, bins=500, range=(0, 3.0), threads=4)
    x = (x[1:] + x[:-1]) / 2.0
    plt.errorbar(x, y, marker=".", yerr=stats.sem(y), linestyle="",)
    plt.savefig("W_hist.png")
