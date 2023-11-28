import pandas as pd
from pathlib import Path
import numpy as np
from utils.utils import gen_RanInlets2, gen_Normal

def main(d_path, s_path, level, nlevel):
    # load data
    df = pd.read_csv(d_path, header=0)
    T1i = df["F1i"].copy().to_numpy()
    T1o = df["F1o"].copy().to_numpy()
    T2i = df["F2i"].copy().to_numpy()
    T2o = df["F2o"].copy().to_numpy()
    m1 = df["F1m"].copy().to_numpy()
    m2 = df["F2m"].copy().to_numpy()
    n = len(df)
    
    # decide random indices
    idx = []
    while(len(idx) < int(n * nlevel)):
        ri = np.random.randint(0, 200)
        if ri not in idx:
            idx.append(ri)
    
    for i in range(n):
        if i in idx:
            T1i[i], T2i[i], m1[i], m2[i] = gen_RanInlets2(T1i[i], m1[i], T2i[i], m2[i], 0.1, 0.3, gen_Normal, T1o[i], T2o[i])
    
    # save data
    dfs1 = df.copy()
    dfs2 = df.copy()
    dfs3 = df.copy()
    # case 1, noisy temperatures
    dfs1["F1i"] = T1i
    dfs1["F2i"] = T2i
    dfs1.to_csv(f"{s_path}/case1/rndata_{level}.csv", index=False, header=True)
    
    # case 2, noisy mass flows
    dfs2["F1m"] = m1
    dfs2["F2m"] = m2
    dfs2.to_csv(f"{s_path}/case2/rndata_{level}.csv", index=False, header=True)
    
    # case 3, mixed
    dfs3["F1i"] = T1i
    dfs3["F2i"] = T2i
    dfs3["F1m"] = m1
    dfs3["F2m"] = m2
    dfs3.to_csv(f"{s_path}/case3/rndata_{level}.csv", index=False, header=True)
    
if __name__ == '__main__':
    levels = ['l', 'm', 'h']
    nlevels = [0.05, 0.1, 0.2]
    d_path = Path("../../py_data/HEXPractice/lumpHEX/rinlet/parallel.csv")
    s_path = Path ("../../py_data/HEXPractice/RN")
    
    for level, nlevel in zip(levels, nlevels):
        main(d_path, s_path, level, nlevel)