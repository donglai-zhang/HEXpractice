import pandas as pd

class GenDataframe:
    def __init__(self) -> None:
        # dataframe for recording data
        self.dfs = pd.DataFrame()
        self.Day = []
        self.UAs = []
        self.T1in = []
        self.T1m = []
        self.T1o = []
        self.T1v = []
        self.T1D = []
        self.T1Re = []
        self.T1Nu = []
        self.T1h = []
        self.T1R = []
        self.T1Cf = []
        self.T1Tau = []
        self.T1pd = []
        self.Sigma1 = []
        self.Rf1 =[]
        self.T2in = []
        self.T2m = []
        self.T2o = []
        self.T2v = []
        self.T2D = []
        self.T2Re = []
        self.T2Nu = []
        self.T2h = []
        self.T2R = []
        self.T2Cf = []
        self.T2Tau = []
        self.T2pd = []
        self.Sigma2 = []
        self.Rf2 =[]
        self.Q = []
    
    def append_Vars(self, day, uas, 
                t1i, t1m, t1v, t1d, t1re, t1nu, t1h, t1r, t1cf, t1tau, t1pd, sigma1, rf1,
                t2i, t2m, t2v, t2d, t2re, t2nu, t2h, t2r, t2cf, t2tau, t2pd, sigma2, rf2):
        self.Day.append(day)
        self.UAs.append(uas)
        self.T1in.append(t1i)
        self.T1m.append(t1m)
        self.T1v.append(t1v)
        self.T1D.append(t1d)
        self.T1Re.append(t1re)
        self.T1Nu.append(t1nu)
        self.T1h.append(t1h)
        self.T1R.append(t1r)
        self.T1Cf.append(t1cf)
        self.T1Tau.append(t1tau)
        self.T1pd.append(t1pd)
        self.Sigma1.append(sigma1)
        self.Rf1.append(rf1)
        self.T2in.append(t2i)
        self.T2m.append(t2m)
        self.T2v.append(t2v)
        self.T2D.append(t2d)
        self.T2Re.append(t2re)
        self.T2Nu.append(t2nu)
        self.T2h.append(t2h)
        self.T2R.append(t2r)
        self.T2Cf.append(t2cf)
        self.T2Tau.append(t2tau)
        self.T2pd.append(t2pd)
        self.Sigma2.append(sigma2)
        self.Rf2.append(rf2)
    
    def append_Outlets(self, t1o, t2o, q):
        self.T1o.append(t1o)
        self.T2o.append(t2o)
        self.Q.append(q)
    
    def export_Vars(self, f_type, dpath):
        # export data frame
        self.dfs["Day"] = self.Day
        self.dfs["F1m"] = self.T1m
        self.dfs["F1i"] = self.T1in
        self.dfs["F1o"] = self.T1o
        self.dfs["F1v"] = self.T1v
        self.dfs["F1D"] = self.T1D
        self.dfs["F1Re"] = self.T1Re
        self.dfs["F1Nu"] = self.T1Nu
        self.dfs["F1h"] = self.T1h
        self.dfs["F1R"] = self.T1R
        self.dfs["F1Cf"] = self.T1Cf
        self.dfs["F1Tau"] = self.T1Tau
        self.dfs["dP1"] = self.T1pd
        self.dfs["Sigma1"] = self.Sigma1
        self.dfs["Rf1"] = self.Rf1
        self.dfs["F2m"] = self.T2m
        self.dfs["F2i"] = self.T2in
        self.dfs["F2o"] = self.T2o
        self.dfs["F2v"] = self.T2v
        self.dfs["F2D"] = self.T2D
        self.dfs["F2Re"] = self.T2Re
        self.dfs["F1Nu"] = self.T1Nu
        self.dfs["F2h"] = self.T2h
        self.dfs["F2R"] = self.T2R
        self.dfs["F2Cf"] = self.T2Cf
        self.dfs["F2Tau"] = self.T2Tau
        self.dfs["dP2"] = self.T2pd
        self.dfs["Sigma2"] = self.Sigma2
        self.dfs["Rf2"] = self.Rf2
        self.dfs["UA"] = self.UAs
        self.dfs["Q"] = self.Q
        
        if f_type == 0:
            self.dfs.to_csv(f"{dpath}/parallel.csv", index=False)
        elif f_type == 1:
            self.dfs.to_csv(f"{dpath}/counter.csv", index=False)