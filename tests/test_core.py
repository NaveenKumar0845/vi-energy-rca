import pandas as pd
from src.data_pipeline import prorate_raw, prepare
from src.demo_data import demo_raw
from src.models import (
    DisputeClassifier,
    add_anomalies,
    canonicalize_target_labels,
    prepare_taxonomy,
)

TAX = pd.DataFrame({
    "Dispute Head": ["EB", "EB", "EB", "DG", "DG", "DG"],
    "Dispute Sub-Category": [
        "High EB Consumption",
        "Incorrect EB Tariffs",
        "Other (Please provide comment under separate Column)",
        "High DG Consumption",
        "Incorrect DG Rate",
        "Other (Please provide comment under separate Column)",
    ],
})


def test_cross_month_proration_reconciles():
    raw = pd.DataFrame([{
        "IP Site ID": "S1",
        "Expense Nature": "EB",
        "Actual Bill From Date": "2023-01-20",
        "Actual Bill To Date": "2023-02-19",
        "Billed Amount (Excl GST)": 31000,
        "Debit Amount (Excl tax)": 3100,
        "Dispute Type": "High EB Consumption",
    }])
    p, _ = prorate_raw(raw)
    assert len(p) == 2
    assert round(p["Prorated Billed Amount (Excl GST)"].sum(), 2) == 31000
    assert round(p.loc[p["Month-Year"].eq("2023-01"), "Prorated Billed Amount (Excl GST)"].iloc[0], 2) == 12000


def test_features_use_prior_history():
    raw = demo_raw().query("`IP Site ID` == 'IN-100001' and `Expense Nature` == 'EB'").head(3)
    p = prepare(raw, "raw").analytical.sort_values("Month-Year")
    assert pd.isna(p.iloc[0]["Site Historical Average"])
    assert p.iloc[1]["Site Historical Average"] == p.iloc[0]["Prorated Billed Amount (Excl GST)"]


def test_classifier_never_crosses_head():
    raw = demo_raw()
    p = prepare(raw, "raw").analytical
    taxonomy = pd.DataFrame({
        "Dispute Head": ["EB"] * 4 + ["DG"] * 4,
        "Dispute Sub-Category": [
            "High EB Consumption",
            "Incorrect EB Tariffs",
            "Duplicate/Retro Billing",
            "Not part of BCC",
            "High DG Consumption",
            "Incorrect DG Rate",
            "Duplicate/Retro Billing",
            "Not part of BCC",
        ],
    })
    t = prepare_taxonomy(taxonomy)
    m = DisputeClassifier(.30).fit(p, t)
    out = m.predict(p)
    eb = set(t.loc[t["Dispute Head"].eq("EB"), "Dispute Sub-Category"])
    dg = set(t.loc[t["Dispute Head"].eq("DG"), "Dispute Sub-Category"])
    for _, r in out.iterrows():
        if r["Predicted Dispute Type"] == "Manual Review Required":
            continue
        assert r["Predicted Dispute Type"] in (eb if r["Expense Nature"] == "EB" else dg)


def test_anomaly_output_exists():
    p = prepare(demo_raw(), "raw").analytical
    out = add_anomalies(p, .10)
    assert {"Is Anomaly", "Anomaly Percentile", "Anomaly Evidence"}.issubset(out.columns)


def test_messy_original_excel_headers_are_canonicalized():
    raw = pd.DataFrame([{
        "IP Site ID": "S1",
        "Expense Nature\nRental/EB/DG/Tax/De-loading": "EB",
        "Actual Bill From Date": "2023-01-01",
        "Actual Bill To Date": "2023-01-31",
        "Billed Amount\n(Excl GST )": 31000,
        "Debit Amount\n(Excl tax)": 3100,
        "Reason for Dispute Categoary": "High EB Consumption",
        "Dispute Type": "High EB Consumption",
    }])
    p, _ = prorate_raw(raw)
    assert p.iloc[0]["Expense Nature"] == "EB"
    assert p.iloc[0]["Prorated Billed Amount (Excl GST)"] == 31000
    assert "Reason for Dispute Category" in p.columns


def test_reason_category_is_preferred_and_safe_aliases_are_normalized():
    taxonomy = prepare_taxonomy(pd.DataFrame({
        "Dispute Head": ["EB", "EB", "EB"],
        "Dispute Sub-Category": [
            "High EB Consumption",
            "Other (Please provide comment under separate Column)",
            "Site Locked / switch-off Billing",
        ],
    }))
    df = pd.DataFrame([
        {"Expense Nature": "EB", "Reason for Dispute Category": "Other", "Dispute Type": "High EB Consumption"},
        {"Expense Nature": "EB", "Reason for Dispute Category": "Site Locked", "Dispute Type": "High EB Consumption"},
    ])
    target, source = canonicalize_target_labels(df, taxonomy)
    assert target.iloc[0] == "Other (Please provide comment under separate Column)"
    assert target.iloc[1] == "Site Locked / switch-off Billing"
    assert source.eq("Reason for Dispute Category").all()


def test_classifier_can_train_eb_and_dg_from_supported_canonical_classes():
    rows = []
    for head, labels in {
        "EB": ["High EB Consumption", "Other"],
        "DG": ["High DG Consumption", "Other"],
    }.items():
        for j, label in enumerate(labels):
            for i in range(8):
                rows.append({
                    "IP Site ID": f"{head}-{j}-{i}",
                    "Expense Nature": head,
                    "Month-Year": f"2023-{(i % 8) + 1:02d}",
                    "Prorated Billed Amount (Excl GST)": 1000 + j * 500 + i * 10,
                    "Prorated Debit Amount (Excl tax)": 100 + j * 20,
                    "Reason for Dispute Category": label,
                    "Dispute Type": label,
                    "Circle Code": "C1" if j == 0 else "C2",
                    "Vendor Code": "V1" if j == 0 else "V2",
                })
    p = prepare(pd.DataFrame(rows), "model-ready").analytical
    model = DisputeClassifier(.30, min_class_rows=5).fit(p, prepare_taxonomy(TAX))
    assert set(model.models) == {"EB", "DG"}
    assert model.label_stats["EB"]["usable_training_rows"] == 16
    assert model.label_stats["DG"]["usable_training_rows"] == 16
