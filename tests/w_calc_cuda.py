import wrapper
import pandas as pd
import numpy as np
import time


wrapper.cuda_properties()

df = pd.read_csv("/mnt/ssd/kincorr/momCorr.csv")

start = time.time()
w = wrapper.cuda_w(df.e_p.to_numpy(), np.deg2rad(
    df.e_theta), np.deg2rad(df.e_phi))
total = time.time() - start
print("W: ", total/df.e_p.to_numpy().size, "Hz")

start = time.time()
w = wrapper.w(df.e_p.to_numpy(), np.deg2rad(
    df.e_theta), np.deg2rad(df.e_phi))
total = time.time() - start
print("W: ", total/df.e_p.to_numpy().size, "Hz")

start = time.time()
q2 = wrapper.cuda_q2(df.e_p.to_numpy(), np.deg2rad(
    df.e_theta), np.deg2rad(df.e_phi))
total = time.time() - start
print("Q2: ", total/df.e_p.to_numpy().size, "Hz")

start = time.time()
q2 = wrapper.q2(df.e_p.to_numpy(), np.deg2rad(
    df.e_theta), np.deg2rad(df.e_phi))
total = time.time() - start
print("Q2: ", total/df.e_p.to_numpy().size, "Hz")
