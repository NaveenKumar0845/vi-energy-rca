from __future__ import annotations
from dataclasses import dataclass, field
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score

NUMERIC = [
    "Prorated Billed Amount (Excl GST)",
    "Prorated Debit Amount (Excl tax)",
    "Prorated Debit Amount (Incl tax)",
    "Billed Amount (Excl GST)",
    "Debit Amount (Excl tax)",
    "Debit Amount (Incl tax)",
    "Debit/Billed Ratio",
    "Bill Duration Days",
    "Proration Days",
    "Previous Month Billing",
    "Site Historical Average",
    "Site Historical Std",
    "Rolling 3M Average",
    "MoM Billing Change %",
    "Difference From Site Average %",
    "Month",
    "Quarter",
    "Invoice Month",
    "Invoice Quarter",
]
CATEGORICAL = ["Circle Code", "Vendor Code", "IP Category"]
TARGET_COLUMN = "Target Dispute Reason"
TARGET_SOURCE_COLUMN = "Target Label Source"


def prepare_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]
    if not {"Dispute Head", "Dispute Sub-Category"}.issubset(x.columns):
        raise ValueError("Dispute taxonomy requires Dispute Head and Dispute Sub-Category")
    x["Dispute Head"] = x["Dispute Head"].astype(str).str.strip().str.upper()
    x["Dispute Sub-Category"] = x["Dispute Sub-Category"].astype(str).str.strip()
    return x.drop_duplicates()


def valid_reasons(taxonomy, head):
    return taxonomy.loc[taxonomy["Dispute Head"].eq(str(head).upper()), "Dispute Sub-Category"].tolist()


def _label_key(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _taxonomy_lookup(taxonomy: pd.DataFrame, head: str) -> dict[str, str]:
    allowed = valid_reasons(taxonomy, head)
    lookup: dict[str, str] = {}
    collisions: set[str] = set()
    for reason in allowed:
        key = _label_key(reason)
        if not key:
            continue
        if key in lookup and lookup[key] != reason:
            collisions.add(key)
        else:
            lookup[key] = reason
    for key in collisions:
        lookup.pop(key, None)

    # Safe legacy aliases observed in the historical Vi workbook. These map only
    # obvious abbreviations to the corresponding approved taxonomy value.
    other = next((r for r in allowed if _label_key(r).startswith("other please provide comment")), None)
    if other:
        lookup.setdefault("other", other)
    locked = next((r for r in allowed if _label_key(r).startswith("site locked switch off billing")), None)
    if locked:
        lookup.setdefault("site locked", locked)
    return lookup


def canonicalize_target_labels(df: pd.DataFrame, taxonomy: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Create a taxonomy-aligned target without overwriting source labels.

    The real workbook contains both a legacy `Dispute Type` field and a more
    taxonomy-aligned `Reason for Dispute Category` field. We prefer the latter,
    then fall back to `Dispute Type`. Only exact normalized matches plus two safe
    legacy aliases (`Other`, `Site Locked`) are accepted; there is no fuzzy or
    semantic guessing.
    """
    tax = prepare_taxonomy(taxonomy)
    target = pd.Series(pd.NA, index=df.index, dtype="string")
    source = pd.Series(pd.NA, index=df.index, dtype="string")
    candidates = [
        c for c in ("Reason for Dispute Category", "Reason for Dispute Categoary", "Dispute Type")
        if c in df.columns
    ]
    if not candidates or "Expense Nature" not in df.columns:
        return target, source

    lookups = {head: _taxonomy_lookup(tax, head) for head in ("EB", "DG")}
    for idx in df.index:
        head = str(df.at[idx, "Expense Nature"]).strip().upper()
        lookup = lookups.get(head)
        if not lookup:
            continue
        for column in candidates:
            mapped = lookup.get(_label_key(df.at[idx, column]))
            if mapped:
                target.at[idx] = mapped
                source.at[idx] = column
                break
    return target, source


@dataclass
class DisputeClassifier:
    threshold: float = 0.60
    min_class_rows: int = 5
    models: dict = field(default_factory=dict)
    taxonomy: pd.DataFrame | None = None
    unavailable_heads: dict = field(default_factory=dict)
    label_stats: dict = field(default_factory=dict)

    def _pipe(self, num, cat):
        tr = []
        if num:
            tr.append(("n", Pipeline([("i", SimpleImputer(strategy="median"))]), num))
        if cat:
            tr.append(("c", Pipeline([
                ("i", SimpleImputer(strategy="most_frequent")),
                ("o", OneHotEncoder(handle_unknown="ignore")),
            ]), cat))
        prep = ColumnTransformer(tr)
        clf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("prep", prep), ("clf", clf)])

    @staticmethod
    def _ensure_features(df, columns):
        x = df.copy()
        for c in columns:
            if c not in x.columns:
                x[c] = np.nan
        return x[columns]

    def fit(self, df, taxonomy):
        self.taxonomy = prepare_taxonomy(taxonomy)
        target, source = canonicalize_target_labels(df, self.taxonomy)
        if target.notna().sum() == 0:
            raise ValueError("Training data needs a taxonomy-aligned dispute label")

        work = df.copy()
        work[TARGET_COLUMN] = target
        work[TARGET_SOURCE_COLUMN] = source
        self.models = {}
        self.unavailable_heads = {}
        self.label_stats = {}

        for head in ["EB", "DG"]:
            head_rows = work[work["Expense Nature"].astype(str).str.upper().eq(head)].copy()
            matched = head_rows[head_rows[TARGET_COLUMN].notna()].copy()
            counts = matched[TARGET_COLUMN].value_counts()
            usable_classes = counts[counts >= self.min_class_rows].index.tolist()
            rare = {str(k): int(v) for k, v in counts[counts < self.min_class_rows].items()}
            s = matched[matched[TARGET_COLUMN].isin(usable_classes)].copy()

            self.label_stats[head] = {
                "head_rows": int(len(head_rows)),
                "taxonomy_aligned_rows": int(len(matched)),
                "taxonomy_aligned_classes": int(counts.size),
                "usable_training_rows": int(len(s)),
                "usable_classes": [str(v) for v in usable_classes],
                "rare_classes_excluded": rare,
                "source_counts": {str(k): int(v) for k, v in matched[TARGET_SOURCE_COLUMN].value_counts().items()},
            }

            if len(s) < 8:
                self.unavailable_heads[head] = f"Only {len(s)} usable taxonomy-aligned training rows"
                continue
            if len(usable_classes) < 2:
                self.unavailable_heads[head] = (
                    f"Fewer than two dispute classes have at least {self.min_class_rows} historical rows"
                )
                continue

            num = [c for c in NUMERIC if c in s and pd.to_numeric(s[c], errors="coerce").notna().any()]
            cat = [c for c in CATEGORICAL if c in s and s[c].notna().any()]
            X = s[num + cat].copy()
            pipe = self._pipe(num, cat)
            pipe.fit(X, s[TARGET_COLUMN].astype(str))
            self.models[head] = (pipe, num, cat)

        if not self.models:
            raise ValueError("No trainable EB/DG classifier could be built")
        return self

    def predict(self, df):
        out = df.copy()
        out["Predicted Dispute Type"] = "Manual Review Required"
        out["Prediction Confidence"] = np.nan
        out["Second Candidate"] = pd.NA
        out["Second Candidate Confidence"] = np.nan
        out["Prediction Status"] = "No eligible model"

        for head, reason in self.unavailable_heads.items():
            mask = out["Expense Nature"].astype(str).str.upper().eq(head)
            out.loc[mask, "Prediction Status"] = f"Manual review: {reason}"

        for head, (pipe, num, cat) in self.models.items():
            mask = out["Expense Nature"].astype(str).str.upper().eq(head)
            if not mask.any():
                continue
            X = self._ensure_features(out.loc[mask], num + cat)
            p = pipe.predict_proba(X)
            classes = pipe.named_steps["clf"].classes_
            order = np.argsort(p, axis=1)[:, ::-1]
            top = order[:, 0]
            prob = p[np.arange(len(p)), top]
            pred = classes[top]
            second = classes[order[:, 1]] if p.shape[1] > 1 else np.array([None] * len(p))
            second_p = p[np.arange(len(p)), order[:, 1]] if p.shape[1] > 1 else np.full(len(p), np.nan)
            ok = prob >= self.threshold
            out.loc[mask, "Predicted Dispute Type"] = np.where(ok, pred, "Manual Review Required")
            out.loc[mask, "Prediction Confidence"] = prob
            out.loc[mask, "Second Candidate"] = second
            out.loc[mask, "Second Candidate Confidence"] = second_p
            out.loc[mask, "Prediction Status"] = np.where(ok, "Auto-predicted", "Low confidence")
        return out


def add_anomalies(df, contamination=0.10):
    out = df.copy()
    candidates = [c for c in NUMERIC if c in out and pd.to_numeric(out[c], errors="coerce").notna().any()]
    use = [c for c in candidates if c not in {"Month", "Quarter", "Invoice Month", "Invoice Quarter"}][:10]
    if not use:
        out["Is Anomaly"] = False
        out["Anomaly Percentile"] = 0.0
        out["Anomaly Evidence"] = "No numeric features"
        return out
    X = out[use].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)
    iso = IsolationForest(contamination=contamination, random_state=42).fit(X)
    score = -iso.score_samples(X)
    out["Is Anomaly"] = iso.predict(X) == -1
    out["Anomaly Percentile"] = pd.Series(score, index=out.index).rank(pct=True)

    def evidence(r):
        parts = []
        for c in ["MoM Billing Change %", "Difference From Site Average %", "Debit/Billed Ratio"]:
            if c in out and pd.notna(r.get(c)):
                parts.append(f"{c}: {r[c]:.1f}")
        return "; ".join(parts[:3]) or "Multivariate billing deviation"

    out["Anomaly Evidence"] = out.apply(evidence, axis=1)
    return out


def _chronological_periods(df: pd.DataFrame) -> tuple[pd.Series, str]:
    # The workbook's source Month column represents the operational record month
    # and is the best available chronology for holdout evaluation. Bill service
    # periods can be retroactive and the observed Invoice Date field has only one
    # month in the supplied workbook, so both are fallbacks rather than defaults.
    if "Record Month" in df:
        record = pd.to_datetime(df["Record Month"], errors="coerce").dt.to_period("M")
        if record.notna().sum() >= max(4, int(len(df) * 0.5)) and record.dropna().nunique() >= 4:
            return record, "Record Month"

    if "Invoice Date" in df:
        invoice = pd.to_datetime(df["Invoice Date"], errors="coerce")
        invoice_periods = invoice.dt.to_period("M")
        if (
            invoice.notna().sum() >= max(4, int(len(df) * 0.5))
            and invoice_periods.dropna().nunique() >= 4
        ):
            return invoice_periods, "Invoice Date"

    return pd.to_datetime(df["Month-Year"], errors="coerce").dt.to_period("M"), "Month-Year"


def evaluate_time_split(df, taxonomy, threshold=0.60):
    periods, basis = _chronological_periods(df)
    valid = periods.notna()
    counts = periods[valid].value_counts().sort_index()
    if len(counts) < 4:
        return {"error": "Need at least four chronological periods", "time_basis": basis}, pd.DataFrame()

    # Select a whole-period cutoff that places roughly 75% of rows in train,
    # rather than 75% of unique months, which is distorted by sparse retro bills.
    cumulative = counts.cumsum()
    target_rows = counts.sum() * 0.75
    cut_candidates = cumulative[cumulative >= target_rows]
    cut = cut_candidates.index[0] if not cut_candidates.empty else counts.index[-2]
    if cut == counts.index[-1]:
        cut = counts.index[-2]

    train = df[periods <= cut]
    test = df[periods > cut]
    if train.empty or test.empty:
        return {"error": "Chronological split produced an empty train or test set", "time_basis": basis}, pd.DataFrame()

    try:
        model = DisputeClassifier(threshold).fit(train, taxonomy)
    except ValueError as exc:
        return {
            "error": str(exc),
            "time_basis": basis,
            "cutoff_period": str(cut),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }, pd.DataFrame()

    pred = model.predict(test)
    actual, source = canonicalize_target_labels(test, taxonomy)
    pred["Actual Canonical Dispute Reason"] = actual
    pred["Actual Label Source"] = source
    eligible = pred[pred["Actual Canonical Dispute Reason"].notna()].copy()
    scored = eligible[eligible["Predicted Dispute Type"].ne("Manual Review Required")].copy()

    m = {
        "time_basis": basis,
        "cutoff_period": str(cut),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "eligible_test_rows": int(len(eligible)),
        "auto_scored_rows": int(len(scored)),
        "coverage": len(scored) / max(1, len(eligible)),
        "manual_review_rate": 1 - (len(scored) / max(1, len(eligible))),
        "trained_heads": sorted(model.models.keys()),
        "label_stats": model.label_stats,
    }
    if not scored.empty:
        y = scored["Actual Canonical Dispute Reason"].astype(str)
        p = scored["Predicted Dispute Type"].astype(str)
        correct = int((y == p).sum())
        m.update(
            accuracy=accuracy_score(y, p),
            macro_f1=f1_score(y, p, average="macro", zero_division=0),
            weighted_f1=f1_score(y, p, average="weighted", zero_division=0),
            effective_accuracy=correct / max(1, len(eligible)),
        )

        per_head = {}
        for head in ("EB", "DG"):
            hs = scored[scored["Expense Nature"].astype(str).str.upper().eq(head)]
            he = eligible[eligible["Expense Nature"].astype(str).str.upper().eq(head)]
            if he.empty:
                continue
            entry = {
                "eligible_rows": int(len(he)),
                "auto_scored_rows": int(len(hs)),
                "coverage": len(hs) / len(he),
            }
            if not hs.empty:
                hy = hs["Actual Canonical Dispute Reason"].astype(str)
                hp = hs["Predicted Dispute Type"].astype(str)
                entry.update(
                    accuracy=accuracy_score(hy, hp),
                    macro_f1=f1_score(hy, hp, average="macro", zero_division=0),
                    weighted_f1=f1_score(hy, hp, average="weighted", zero_division=0),
                )
            per_head[head] = entry
        m["per_head"] = per_head
    return m, pred
