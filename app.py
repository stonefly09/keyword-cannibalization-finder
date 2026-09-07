import json
import re
import tempfile
from datetime import date, timedelta
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]

st.set_page_config(page_title="Keyword Cannibalization Finder", page_icon="🔎", layout="wide")


def normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def weighted_position(group: pd.DataFrame) -> float:
    weights = group["impressions"].clip(lower=0)
    if weights.sum() <= 0:
        return float(group["position"].mean())
    return float((group["position"] * weights).sum() / weights.sum())


def aggregate_query_page(df: pd.DataFrame) -> pd.DataFrame:
    required = {"query", "page", "clicks", "impressions", "ctr", "position"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    work = df.copy()
    work["query"] = work["query"].astype(str).map(normalize_query)
    work["page"] = work["page"].astype(str).str.strip()
    for col in ["clicks", "impressions", "ctr", "position"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    rows = []
    for (query, page), group in work.groupby(["query", "page"], dropna=False):
        impressions = float(group["impressions"].sum())
        clicks = float(group["clicks"].sum())
        rows.append(
            {
                "query": query,
                "page": page,
                "clicks": clicks,
                "impressions": impressions,
                "ctr": clicks / impressions if impressions else 0.0,
                "position": weighted_position(group),
            }
        )
    return pd.DataFrame(rows)


def score_candidate(total_impressions: float, second_share: float, pos1: float, pos2: float) -> int:
    # Heuristic score, deliberately conservative. It is not a Google metric.
    score = 0.0
    score += 50 * min(second_share / 0.50, 1.0)
    score += 20 * min(total_impressions / 1000.0, 1.0)
    if pos1 <= 20 and pos2 <= 20:
        score += 15
    if abs(pos1 - pos2) <= 5:
        score += 15
    return int(round(min(score, 100)))


def severity_from_score(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def recommendation(primary_pos: float, secondary_pos: float, secondary_share: float) -> str:
    if primary_pos <= 10 and secondary_pos > 10:
        return "Protect the stronger URL; retarget, consolidate, or internally de-emphasize the secondary page after checking intent."
    if primary_pos <= 10 and secondary_pos <= 10:
        return "Both URLs already perform. Check whether Google is rewarding distinct intent before merging; differentiate targeting if overlap is accidental."
    if secondary_share >= 0.30:
        return "Strong traffic split. Compare intent/content, choose a primary URL, then consider consolidation, canonicalization, or clearer internal linking."
    return "Review search intent and page purpose. If both pages target the same intent, strengthen one primary URL and reduce overlap."


def detect_cannibalization(
    df: pd.DataFrame,
    min_total_impressions: int = 50,
    min_secondary_share: float = 0.10,
    min_secondary_impressions: int = 10,
    exclude_regex: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qp = aggregate_query_page(df)

    if exclude_regex.strip():
        try:
            pattern = re.compile(exclude_regex, flags=re.I)
            qp = qp[~qp["query"].str.contains(pattern, na=False)]
        except re.error as exc:
            raise ValueError(f"Invalid exclude regex: {exc}") from exc

    summary_rows = []
    detail_rows = []

    for query, group in qp.groupby("query"):
        group = group.sort_values(["impressions", "clicks"], ascending=False).reset_index(drop=True)
        if group["page"].nunique() < 2:
            continue

        total_imp = float(group["impressions"].sum())
        total_clicks = float(group["clicks"].sum())
        if total_imp < min_total_impressions:
            continue

        primary = group.iloc[0]
        secondary = group.iloc[1]
        secondary_share = float(secondary["impressions"] / total_imp) if total_imp else 0.0

        if secondary["impressions"] < min_secondary_impressions or secondary_share < min_secondary_share:
            continue

        score = score_candidate(total_imp, secondary_share, float(primary["position"]), float(secondary["position"]))
        severity = severity_from_score(score)

        summary_rows.append(
            {
                "query": query,
                "severity": severity,
                "score": score,
                "urls": int(group["page"].nunique()),
                "clicks": round(total_clicks, 0),
                "impressions": round(total_imp, 0),
                "primary_url": primary["page"],
                "primary_position": round(float(primary["position"]), 2),
                "primary_impression_share": round(float(primary["impressions"] / total_imp), 4),
                "competing_url": secondary["page"],
                "competing_position": round(float(secondary["position"]), 2),
                "competing_impression_share": round(secondary_share, 4),
                "recommendation": recommendation(float(primary["position"]), float(secondary["position"]), secondary_share),
            }
        )

        for _, row in group.iterrows():
            detail_rows.append(
                {
                    "query": query,
                    "severity": severity,
                    "score": score,
                    "page": row["page"],
                    "clicks": round(float(row["clicks"]), 0),
                    "impressions": round(float(row["impressions"]), 0),
                    "ctr": float(row["ctr"]),
                    "position": round(float(row["position"]), 2),
                    "impression_share": float(row["impressions"] / total_imp) if total_imp else 0.0,
                }
            )

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)

    if not summary.empty:
        severity_order = pd.Categorical(summary["severity"], categories=["High", "Medium", "Low"], ordered=True)
        summary = summary.assign(_sev=severity_order).sort_values(["_sev", "score", "impressions"], ascending=[True, False, False]).drop(columns="_sev")

    return summary, details


def get_service(service_account_info: dict):
    creds = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    return build("searchconsole", "v1", credentials=creds, cache_discovery=False)


def list_properties(service) -> list[str]:
    response = service.sites().list().execute()
    return [entry["siteUrl"] for entry in response.get("siteEntry", [])]


def fetch_gsc_data(service, site_url: str, start_date: date, end_date: date) -> pd.DataFrame:
    all_rows = []
    start_row = 0
    row_limit = 25000

    while True:
        body = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "dimensions": ["query", "page"],
            "type": "web",
            "rowLimit": row_limit,
            "startRow": start_row,
            "dataState": "final",
        }
        response = service.searchanalytics().query(siteUrl=site_url, body=body).execute()
        rows = response.get("rows", [])
        if not rows:
            break

        for row in rows:
            keys = row.get("keys", ["", ""])
            all_rows.append(
                {
                    "query": keys[0] if len(keys) > 0 else "",
                    "page": keys[1] if len(keys) > 1 else "",
                    "clicks": row.get("clicks", 0),
                    "impressions": row.get("impressions", 0),
                    "ctr": row.get("ctr", 0),
                    "position": row.get("position", 0),
                }
            )

        if len(rows) < row_limit:
            break
        start_row += row_limit

    return pd.DataFrame(all_rows)


def format_summary_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    out["primary_impression_share"] = (out["primary_impression_share"] * 100).round(1).astype(str) + "%"
    out["competing_impression_share"] = (out["competing_impression_share"] * 100).round(1).astype(str) + "%"
    return out


st.title("Keyword Cannibalization Finder")
st.caption("Find likely SEO cannibalization candidates from Google Search Console query × page data.")

with st.expander("How this tool decides what to flag", expanded=False):
    st.markdown(
        """
        A query appearing for multiple URLs is **not automatically a problem**. This tool only flags candidates when the second URL has meaningful visibility.

        The score uses:
        - the competing URL's share of query impressions,
        - total query demand inside your GSC data,
        - whether both URLs rank in the top 20,
        - and how close their average positions are.

        Treat the result as an SEO review queue, not as an automatic instruction to merge pages.
        """
    )

source = st.radio("Data source", ["Google Search Console API", "Upload CSV"], horizontal=True)

df = pd.DataFrame()

if source == "Google Search Console API":
    st.subheader("1. Connect Search Console")
    uploaded_key = st.file_uploader("Upload a Google service-account JSON key", type=["json"])
    st.info("The service-account email must be added as a user on the Search Console property. The app only requests read-only Search Console access.")

    if uploaded_key:
        try:
            key_info = json.load(uploaded_key)
            service = get_service(key_info)
            properties = list_properties(service)
            if not properties:
                st.warning("No Search Console properties are available to this service account.")
            else:
                site_url = st.selectbox("Search Console property", properties)
                c1, c2 = st.columns(2)
                default_end = date.today() - timedelta(days=3)
                default_start = default_end - timedelta(days=27)
                start_date = c1.date_input("Start date", value=default_start)
                end_date = c2.date_input("End date", value=default_end)

                if st.button("Fetch GSC data", type="primary"):
                    if start_date > end_date:
                        st.error("Start date must be before end date.")
                    else:
                        with st.spinner("Fetching Search Console rows..."):
                            df = fetch_gsc_data(service, site_url, start_date, end_date)
                        st.session_state["gsc_df"] = df
                        st.success(f"Fetched {len(df):,} query × page rows.")
        except Exception as exc:
            st.error(f"Could not connect to Search Console: {exc}")

    if "gsc_df" in st.session_state:
        df = st.session_state["gsc_df"]

else:
    st.subheader("1. Upload query × page data")
    st.write("Required columns: `query`, `page`, `clicks`, `impressions`, `ctr`, `position`.")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            st.success(f"Loaded {len(df):,} rows.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

if not df.empty:
    st.subheader("2. Detection settings")
    c1, c2, c3 = st.columns(3)
    min_total_impressions = c1.number_input("Minimum total impressions/query", min_value=1, value=50, step=10)
    min_secondary_share_pct = c2.slider("Minimum competing URL impression share", min_value=1, max_value=50, value=10, step=1)
    min_secondary_impressions = c3.number_input("Minimum competing URL impressions", min_value=1, value=10, step=5)
    exclude_regex = st.text_input("Optional exclude-query regex", placeholder="brandname|login|support")

    try:
        summary, details = detect_cannibalization(
            df,
            min_total_impressions=int(min_total_impressions),
            min_secondary_share=min_secondary_share_pct / 100,
            min_secondary_impressions=int(min_secondary_impressions),
            exclude_regex=exclude_regex,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("3. Cannibalization candidates")
    if summary.empty:
        st.success("No candidates matched the current thresholds.")
    else:
        high_count = int((summary["severity"] == "High").sum())
        medium_count = int((summary["severity"] == "Medium").sum())
        low_count = int((summary["severity"] == "Low").sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates", f"{len(summary):,}")
        m2.metric("High", f"{high_count:,}")
        m3.metric("Medium", f"{medium_count:,}")
        m4.metric("Low", f"{low_count:,}")

        display_cols = [
            "query", "severity", "score", "urls", "clicks", "impressions",
            "primary_url", "primary_position", "primary_impression_share",
            "competing_url", "competing_position", "competing_impression_share",
            "recommendation",
        ]
        st.dataframe(format_summary_for_display(summary)[display_cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Download candidate summary CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="keyword_cannibalization_candidates.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download URL-level details CSV",
            data=details.to_csv(index=False).encode("utf-8"),
            file_name="keyword_cannibalization_details.csv",
            mime="text/csv",
        )

        st.subheader("Inspect one query")
        selected_query = st.selectbox("Query", summary["query"].tolist())
        query_details = details[details["query"] == selected_query].copy().sort_values("impressions", ascending=False)
        if not query_details.empty:
            query_details["ctr"] = (query_details["ctr"] * 100).round(2).astype(str) + "%"
            query_details["impression_share"] = (query_details["impression_share"] * 100).round(1).astype(str) + "%"
            st.dataframe(query_details, use_container_width=True, hide_index=True)

st.divider()
st.caption("MVP v1: query × page overlap scoring. Next upgrade: daily URL-switching detection, intent similarity, title/H1 comparison, and automated fix recommendations.")
