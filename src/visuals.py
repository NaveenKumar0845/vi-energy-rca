import pandas as pd
import plotly.express as px

def monthly_billing(df):
    x=df.groupby("Month-Year",as_index=False)["Prorated Billed Amount (Excl GST)"].sum()
    return px.line(x,x="Month-Year",y="Prorated Billed Amount (Excl GST)",markers=True,title="Monthly Prorated Billing")

def disputes(df):
    col="Predicted Dispute Type" if "Predicted Dispute Type" in df else "Dispute Type"
    x=df[col].value_counts().rename_axis(col).reset_index(name="Count")
    return px.bar(x,x=col,y="Count",title="Dispute Distribution")

def anomalies(df):
    y="MoM Billing Change %" if "MoM Billing Change %" in df else "Prorated Billed Amount (Excl GST)"
    return px.scatter(df,x="Prorated Billed Amount (Excl GST)",y=y,color="Is Anomaly",hover_data=[c for c in ["IP Site ID","Month-Year","Expense Nature","Predicted Dispute Type"] if c in df],title="Billing Anomaly View")
