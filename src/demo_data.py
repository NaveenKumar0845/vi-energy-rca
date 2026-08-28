from __future__ import annotations
import numpy as np
import pandas as pd

def demo_raw(seed=42):
    rng=np.random.default_rng(seed); rows=[]
    reasons={"EB":["High EB Consumption","Incorrect EB Tariffs","Duplicate/Retro Billing","Not part of BCC"],
             "DG":["High DG Consumption","Incorrect DG Rate","Duplicate/Retro Billing","Not part of BCC"]}
    for s in range(1,9):
        site=f"IN-{100000+s}"; circle=["DL","MH","KA","GJ"][s%4]
        for month in range(1,13):
            start=pd.Timestamp(2023,month,1); end=start+pd.offsets.MonthEnd(0)
            for exp in ["EB","DG"]:
                reason=reasons[exp][(s+month+(exp=="DG"))%4]; base=(18000 if exp=="EB" else 9000)+s*250+month*120
                billed=base+rng.normal(0,500); ratio=.08
                if reason.startswith("High"): billed+=6000; ratio=.12
                elif reason.startswith("Incorrect"): billed+=3500; ratio=.35
                elif reason=="Duplicate/Retro Billing": billed+=5500; ratio=.72
                elif reason=="Not part of BCC": ratio=.95
                debit=max(0,billed*ratio+rng.normal(0,80))
                rows.append({"IP Site ID":site,"Circle Code":circle,"Vendor Code":["INDUS","ATC"][s%2],"Invoice Date":end,
                    "Actual Bill From Date":start,"Actual Bill To Date":end,"Expense Nature":exp,"Billed Amount (Excl GST)":round(billed,2),
                    "Debit Amount (Excl tax)":round(debit,2),"Debit Amount (Incl tax)":round(debit*1.18,2),"Dispute Type":reason})
    return pd.DataFrame(rows)

def demo_kpis(seed=7):
    rng=np.random.default_rng(seed); rows=[]
    for s in range(1,9):
        site=f"IN-{100000+s}"
        for month in range(1,13):
            rows.append({"IP Site ID":site,"Month-Year":f"2023-{month:02d}","RSRP (dBm)":round(-98-(s%4)*2+rng.normal(0,2),2),
                         "RSRQ (dB)":round(-10-(s%3)+rng.normal(0,1),2),"SINR (dB)":round(12-(s%4)+rng.normal(0,1.2),2),
                         "DL Throughput (Mbps)":round(42+rng.normal(0,5),2),"PRB Utilization (%)":round(58+(s%4)*5+month*.7+rng.normal(0,3),2),
                         "Availability (%)":round(99.55-(s%5)*.08+rng.normal(0,.06),3),"Alarm Count":int(rng.poisson(.5))})
    return pd.DataFrame(rows)
