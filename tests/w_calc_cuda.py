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
print("cuda W: ", df.e_p.to_numpy().size/total/1E6, "MHz")

start = time.time()
for i in range(100):
    w = wrapper.cuda_w(df.e_p.to_numpy(), np.deg2rad(
        df.e_theta), np.deg2rad(df.e_phi))
total = time.time() - start
print("W: ", (df.e_p.to_numpy().size/total)/1E6, "MHz")

start = time.time()
q2 = wrapper.cuda_q2(df.e_p.to_numpy(), np.deg2rad(
    df.e_theta), np.deg2rad(df.e_phi))
total = time.time() - start
print("Q2: ", (df.e_p.to_numpy().size/total)/1E6, "MHz")

start = time.time()
q2 = wrapper.q2(df.e_p.to_numpy(), np.deg2rad(
    df.e_theta), np.deg2rad(df.e_phi))
total = time.time() - start
print("Q2: ", (df.e_p.to_numpy().size/total)/1E6, "MHz")

# image = wrapper.jacobi(512, 512, 0.0543, 1.0, 0.00000001, 50000)
