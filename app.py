import re
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

SCOPES = "https://www.googleapis.com/auth/webmasters.readonly"
GSC_API_BASE = "https://www.googleapis.com/webmasters/v3"

st.set_page_config(
    page_title="Keyword Cannibalization Finder",
    page_icon="🔎",
    layout="wide",
)


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


def score_candidate(
    total_impressions: float,
    second_share: float,
    pos1: float,
    pos2: float,
) -> int:
    # Heuristic score for prioritization. This is not a Google metric.
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
        return (
            "Protect the stronger URL. Check intent, then retarget, consolidate, "
            "or internally de-emphasize the secondary page if the overlap is accidental."
        )
    if primary_pos <= 10 and secondary_pos <= 10:
        return (
            "Both URLs perform well. Confirm whether they satisfy different intents before "
            "merging; differentiate targeting if the overlap is accidental."
        )
    if secondary_share >= 0.30:
        return (
            "Strong visibility split. Compare intent/content, choose a primary URL when "
            "appropriate, then consider consolidation or clearer internal linking."
        )
    return (
        "Review search intent and page purpose. If both pages target the same intent, "
        "strengthen one primary URL and reduce overlap."
    )


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
        group = group.sort_values(
            ["impressions", "clicks"], ascending=False
        ).reset_index(drop=True)

        if group["page"].nunique() < 2:
            continue

        total_imp = float(group["impressions"].sum())
        total_clicks = float(group["clicks"].sum())

        if total_imp < min_total_impressions:
            continue

        primary = group.iloc[0]
        secondary = group.iloc[1]
        secondary_share = (
            float(secondary["impressions"] / total_imp) if total_imp else 0.0
        )

        if (
            secondary["impressions"] < min_secondary_impressions
            or secondary_share < min_secondary_share
        ):
            continue

        score = score_candidate(
            total_imp,
            secondary_share,
            float(primary["position"]),
            float(secondary["position"]),
        )
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
                "primary_impression_share": round(
                    float(primary["impressions"] / total_imp), 4
                ),
                "competing_url": secondary["page"],
                "competing_position": round(float(secondary["position"]), 2),
                "competing_impression_share": round(secondary_share, 4),
                "recommendation": recommendation(
                    float(primary["position"]),
                    float(secondary["position"]),
                    secondary_share,
                ),
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
                    "impression_share": (
                        float(row["impressions"] / total_imp) if total_imp else 0.0
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)

    if not summary.empty:
        severity_order = pd.Categorical(
            summary["severity"],
            categories=["High", "Medium", "Low"],
            ordered=True,
        )
        summary = (
            summary.assign(_sev=severity_order)
            .sort_values(
                ["_sev", "score", "impressions"],
                ascending=[True, False, False],
            )
            .drop(columns="_sev")
        )

    return summary, details


def format_summary_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    out = summary.copy()
    out["primary_impression_share"] = (
        (out["primary_impression_share"] * 100).round(1).astype(str) + "%"
    )
    out["competing_impression_share"] = (
        (out["competing_impression_share"] * 100).round(1).astype(str) + "%"
    )
    return out


def auth_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }


def api_error(response: requests.Response) -> RuntimeError:
    if response.status_code == 401:
        return RuntimeError(
            "Your Google access token has expired or is no longer valid. "
            "Log out, then sign in with Google again."
        )
    if response.status_code == 403:
        return RuntimeError(
            "Google denied access. Make sure this Google account has access to the "
            "Search Console property and that the webmasters.readonly scope is configured."
        )

    try:
        payload = response.json()
        message = payload.get("error", {}).get("message")
    except Exception:
        message = None

    return RuntimeError(
        message or f"Search Console API returned HTTP {response.status_code}."
    )


def list_properties(access_token: str) -> list[dict]:
    response = requests.get(
        f"{GSC_API_BASE}/sites",
        headers=auth_headers(access_token),
        timeout=30,
    )
    if not response.ok:
        raise api_error(response)

    properties = response.json().get("siteEntry", [])
    return sorted(properties, key=lambda item: item.get("siteUrl", ""))


def fetch_gsc_data(
    access_token: str,
    site_url: str,
    start_date: date,
    end_date: date,
    search_type: str = "web",
) -> pd.DataFrame:
    all_rows = []
    start_row = 0
    row_limit = 25_000
    encoded_site = quote(site_url, safe="")
    endpoint = f"{GSC_API_BASE}/sites/{encoded_site}/searchAnalytics/query"

    while True:
        body = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "dimensions": ["query", "page"],
            "type": search_type,
            "rowLimit": row_limit,
            "startRow": start_row,
            "dataState": "final",
        }

        response = requests.post(
            endpoint,
            headers={**auth_headers(access_token), "Content-Type": "application/json"},
            json=body,
            timeout=90,
        )
        if not response.ok:
            raise api_error(response)

        rows = response.json().get("rows", [])
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


def login_screen() -> None:
    st.title("Keyword Cannibalization Finder")
    st.caption(
        "Connect your Google Search Console account and find queries where multiple URLs "
        "are competing for meaningful visibility."
    )
    st.info(
        "Sign in with the Google account that already has access to your Search Console property. "
        "The app requests read-only Search Console access."
    )
    if st.button("Sign in with Google", type="primary"):
        st.login()


# --- Authentication gate ---------------------------------------------------
try:
    logged_in = bool(st.user.is_logged_in)
except Exception:
    st.error(
        "Authentication is not configured correctly. Check the [auth] settings in "
        "Streamlit Community Cloud → App settings → Secrets."
    )
    st.stop()

if not logged_in:
    login_screen()
    st.stop()

# Never print or display st.user.tokens. Access tokens are sensitive credentials.
try:
    access_token = st.user.tokens["access"]
except Exception:
    st.error(
        "Google login succeeded, but no access token is available. In Streamlit Secrets, "
        "make sure `expose_tokens = [\"access\"]` is inside `[auth]`, then log out and sign in again."
    )
    if st.button("Log out"):
        st.logout()
    st.stop()

# --- App -------------------------------------------------------------------
st.title("Keyword Cannibalization Finder")

user_email = getattr(st.user, "email", "Google user")
header_left, header_right = st.columns([5, 1])
header_left.caption(f"Signed in as {user_email}")
if header_right.button("Log out"):
    st.logout()

with st.expander("How this tool decides what to flag", expanded=False):
    st.markdown(
        """
        A query appearing for multiple URLs is **not automatically cannibalization**.
        This tool only flags candidates when the competing URL has meaningful visibility.

        The score considers:
        - competing URL impression share,
        - total query impressions,
        - whether both URLs rank in the top 20,
        - and how close their average positions are.

        Treat the output as an SEO review queue, not an automatic instruction to merge pages.
        """
    )

source = st.radio(
    "Data source",
    ["Google Search Console", "Upload query × page CSV"],
    horizontal=True,
)

df = pd.DataFrame()

if source == "Google Search Console":
    st.subheader("1. Select Search Console data")

    try:
        properties = list_properties(access_token)
    except Exception as exc:
        st.error(str(exc))
        if "expired" in str(exc).lower() or "valid" in str(exc).lower():
            if st.button("Log out and sign in again"):
                st.logout()
        st.stop()

    if not properties:
        st.warning(
            "No Search Console properties were returned for this Google account. "
            "Verify that the signed-in account has Search Console access."
        )
        st.stop()

    property_lookup = {
        p.get("siteUrl", ""): p.get("permissionLevel", "") for p in properties
    }
    site_url = st.selectbox("Search Console property", list(property_lookup.keys()))
    st.caption(f"Permission: {property_lookup.get(site_url, 'unknown')}")

    c1, c2, c3 = st.columns(3)
    default_end = date.today() - timedelta(days=3)
    default_start = default_end - timedelta(days=27)
    start_date = c1.date_input("Start date", value=default_start)
    end_date = c2.date_input("End date", value=default_end)
    search_type = c3.selectbox(
        "Search type",
        ["web", "image", "video", "news"],
        index=0,
    )

    if st.button("Fetch Search Console data", type="primary"):
        if start_date > end_date:
            st.error("Start date must be before or equal to end date.")
        else:
            with st.spinner("Fetching query × page rows from Search Console..."):
                try:
                    fetched = fetch_gsc_data(
                        access_token,
                        site_url,
                        start_date,
                        end_date,
                        search_type,
                    )
                except Exception as exc:
                    st.error(str(exc))
                    st.stop()

            st.session_state["gsc_df"] = fetched
            st.session_state["gsc_context"] = {
                "site_url": site_url,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "search_type": search_type,
            }
            st.success(f"Fetched {len(fetched):,} query × page rows.")

    context = st.session_state.get("gsc_context", {})
    if context.get("site_url") == site_url and "gsc_df" in st.session_state:
        df = st.session_state["gsc_df"]
        st.caption(
            f"Loaded: {context.get('start_date')} → {context.get('end_date')} · "
            f"{context.get('search_type')}"
        )

else:
    st.subheader("1. Upload query × page data")
    st.write(
        "Required columns: `query`, `page`, `clicks`, `impressions`, `ctr`, `position`."
    )
    st.warning(
        "A normal GSC export creates separate Queries.csv and Pages.csv files. Those cannot "
        "be reliably merged for cannibalization analysis. Use a true query × page export."
    )
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
    min_total_impressions = c1.number_input(
        "Minimum total impressions/query",
        min_value=1,
        value=50,
        step=10,
    )
    min_secondary_share_pct = c2.slider(
        "Minimum competing URL impression share",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )
    min_secondary_impressions = c3.number_input(
        "Minimum competing URL impressions",
        min_value=1,
        value=10,
        step=5,
    )
    exclude_regex = st.text_input(
        "Optional exclude-query regex",
        placeholder="brandname|login|support",
    )

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
            "query",
            "severity",
            "score",
            "urls",
            "clicks",
            "impressions",
            "primary_url",
            "primary_position",
            "primary_impression_share",
            "competing_url",
            "competing_position",
            "competing_impression_share",
            "recommendation",
        ]

        st.dataframe(
            format_summary_for_display(summary)[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download candidate summary CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="keyword_cannibalization_candidates.csv",
            mime="text/csv",
        )
        d2.download_button(
            "Download URL-level details CSV",
            data=details.to_csv(index=False).encode("utf-8"),
            file_name="keyword_cannibalization_details.csv",
            mime="text/csv",
        )

        st.subheader("Inspect one query")
        selected_query = st.selectbox("Query", summary["query"].tolist())
        query_details = (
            details[details["query"] == selected_query]
            .copy()
            .sort_values("impressions", ascending=False)
        )

        if not query_details.empty:
            query_details["ctr"] = (
                (query_details["ctr"] * 100).round(2).astype(str) + "%"
            )
            query_details["impression_share"] = (
                (query_details["impression_share"] * 100).round(1).astype(str) + "%"
            )
            st.dataframe(query_details, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "MVP v2: Google OAuth + Search Console query × page overlap scoring. "
    "Next upgrade: URL-switching detection and intent similarity."
)
