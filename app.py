import base64
import hashlib
import hmac
import io
import json
import re
import secrets
import time
from datetime import date, timedelta
from urllib.parse import quote, urlencode

import pandas as pd
import requests
import streamlit as st
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]
GSC_API_BASE = "https://www.googleapis.com/webmasters/v3"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
REQUIRED_COLUMNS = {"query", "page", "clicks", "impressions", "ctr", "position"}

st.set_page_config(
    page_title="Keyword Cannibalization Finder",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
    .seo-card {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 14px;
        padding: 16px 18px;
        background: rgba(250,250,250,.02);
        min-height: 170px;
    }
    .seo-card h4 {margin: 0 0 10px 0; font-size: 1rem;}
    .seo-url {font-size: .88rem; word-break: break-all; line-height: 1.35;}
    .seo-kpi {font-size: 1.55rem; font-weight: 700; margin: 4px 0;}
    .seo-muted {opacity: .72; font-size: .86rem;}
    .score-high {font-weight: 700;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.14); padding: 10px 12px; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "top queries": "query",
        "queries": "query",
        "keyword": "query",
        "keywords": "query",
        "top pages": "page",
        "pages": "page",
        "url": "page",
        "urls": "page",
        "clicks": "clicks",
        "impressions": "impressions",
        "ctr": "ctr",
        "position": "position",
        "average position": "position",
        "avg position": "position",
    }
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in aliases:
            rename[col] = aliases[key]
    return df.rename(columns=rename)


def parse_ctr(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    pct_mask = text.str.endswith("%")
    numeric = pd.to_numeric(text.str.rstrip("%"), errors="coerce")
    numeric.loc[pct_mask] = numeric.loc[pct_mask] / 100
    return numeric.fillna(0.0)


def weighted_position(group: pd.DataFrame) -> float:
    weights = group["impressions"].clip(lower=0)
    if weights.sum() <= 0:
        return float(group["position"].mean())
    return float((group["position"] * weights).sum() / weights.sum())


def aggregate_query_page(df: pd.DataFrame) -> pd.DataFrame:
    work = normalize_csv_columns(df.copy())
    missing = REQUIRED_COLUMNS.difference(work.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing)) + ". "
            "The file must contain query + page in the same row."
        )

    work["query"] = work["query"].astype(str).map(normalize_query)
    work["page"] = work["page"].astype(str).str.strip()
    work["clicks"] = pd.to_numeric(work["clicks"], errors="coerce").fillna(0)
    work["impressions"] = pd.to_numeric(work["impressions"], errors="coerce").fillna(0)
    work["ctr"] = parse_ctr(work["ctr"])
    work["position"] = pd.to_numeric(work["position"], errors="coerce").fillna(0)

    work = work[(work["query"] != "") & (work["page"] != "")]

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
    """Heuristic priority score. This is not a Google metric."""
    score = 0.0
    score += 45 * min(second_share / 0.50, 1.0)
    score += 20 * min(total_impressions / 1000.0, 1.0)
    if pos1 <= 20 and pos2 <= 20:
        score += 20
    if abs(pos1 - pos2) <= 5:
        score += 15
    return int(round(min(score, 100)))


def severity_from_score(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def conflict_type(primary_pos: float, secondary_pos: float, secondary_share: float) -> str:
    if primary_pos <= 20 and secondary_pos <= 20 and secondary_share >= 0.20:
        return "Likely conflict"
    if abs(primary_pos - secondary_pos) <= 5 and secondary_share >= 0.10:
        return "Possible conflict"
    return "Overlap to review"


def recommendation(primary_pos: float, secondary_pos: float, secondary_share: float) -> str:
    if primary_pos <= 10 and secondary_pos <= 10 and secondary_share >= 0.20:
        return (
            "Compare intent first. If both pages answer the same intent, choose one primary URL and "
            "consider consolidation/redirect; otherwise differentiate titles, headings and internal anchors."
        )
    if primary_pos <= 10 and secondary_pos > 10:
        return (
            "Protect the stronger URL. Retarget the secondary page to a distinct intent and strengthen "
            "internal links toward the primary page if the overlap is accidental."
        )
    if secondary_share >= 0.30:
        return (
            "Visibility is strongly split. Review content similarity, intent and internal linking; "
            "consolidate only when both URLs serve essentially the same search intent."
        )
    return (
        "Review the overlap before changing anything. Differentiate targeting if the pages serve "
        "different intents; consolidate only if they are genuinely redundant."
    )


def compact_url(url: str, max_len: int = 72) -> str:
    text = str(url).strip()
    text = re.sub(r"^https?://", "", text)
    text = text.rstrip("/")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def score_breakdown(total_impressions: float, second_share: float, pos1: float, pos2: float) -> dict:
    share_points = 45 * min(second_share / 0.50, 1.0)
    demand_points = 20 * min(total_impressions / 1000.0, 1.0)
    top20_points = 20 if pos1 <= 20 and pos2 <= 20 else 0
    proximity_points = 15 if abs(pos1 - pos2) <= 5 else 0
    return {
        "Competing URL share": round(share_points, 1),
        "Keyword demand": round(demand_points, 1),
        "Both URLs in Top 20": top20_points,
        "Rankings within 5 positions": proximity_points,
    }


def detect_cannibalization(
    df: pd.DataFrame,
    min_total_impressions: int = 50,
    min_secondary_share: float = 0.10,
    min_secondary_impressions: int = 10,
    max_position: float = 100.0,
    exclude_regex: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qp = aggregate_query_page(df)
    qp = qp[qp["position"] <= max_position]

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
        primary_share = float(primary["impressions"] / total_imp) if total_imp else 0.0
        secondary_share = float(secondary["impressions"] / total_imp) if total_imp else 0.0

        if secondary["impressions"] < min_secondary_impressions or secondary_share < min_secondary_share:
            continue

        primary_pos = float(primary["position"])
        secondary_pos = float(secondary["position"])
        score = score_candidate(total_imp, secondary_share, primary_pos, secondary_pos)
        severity = severity_from_score(score)

        summary_rows.append(
            {
                "query": query,
                "severity": severity,
                "conflict_type": conflict_type(primary_pos, secondary_pos, secondary_share),
                "score": score,
                "urls": int(group["page"].nunique()),
                "clicks": round(total_clicks, 0),
                "impressions": round(total_imp, 0),
                "primary_url": primary["page"],
                "primary_position": round(primary_pos, 2),
                "primary_impression_share": round(primary_share, 4),
                "competing_url": secondary["page"],
                "competing_position": round(secondary_pos, 2),
                "competing_impression_share": round(secondary_share, 4),
                "position_gap": round(abs(primary_pos - secondary_pos), 2),
                "recommendation": recommendation(primary_pos, secondary_pos, secondary_share),
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
        severity_order = pd.Categorical(
            summary["severity"], categories=["High", "Medium", "Low"], ordered=True
        )
        summary = (
            summary.assign(_sev=severity_order)
            .sort_values(["_sev", "score", "impressions"], ascending=[True, False, False])
            .drop(columns="_sev")
            .reset_index(drop=True)
        )

    return summary, details


def format_summary_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if out.empty:
        return out
    for col in ["primary_impression_share", "competing_impression_share"]:
        out[col] = (out[col] * 100).round(1).astype(str) + "%"
    return out


def auth_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}


def api_error(response: requests.Response) -> RuntimeError:
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message")
    except Exception:
        message = None

    if response.status_code == 401:
        return RuntimeError("Authentication expired or is invalid. Reconnect and try again.")
    if response.status_code == 403:
        return RuntimeError(
            "Google denied access. Confirm this account/service account has access to the Search Console property."
        )
    return RuntimeError(message or f"Search Console API returned HTTP {response.status_code}.")


def list_properties(access_token: str) -> list[dict]:
    response = requests.get(
        f"{GSC_API_BASE}/sites", headers=auth_headers(access_token), timeout=30
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


def service_account_token(uploaded_file) -> tuple[str, str]:
    try:
        info = json.loads(uploaded_file.getvalue().decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not read the JSON key: {exc}") from exc

    if info.get("type") != "service_account":
        raise ValueError("This does not appear to be a Google service-account JSON key.")

    try:
        credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        credentials.refresh(GoogleAuthRequest())
    except Exception as exc:
        raise ValueError(f"Could not authenticate the service account: {exc}") from exc

    return credentials.token, info.get("client_email", "Unknown service account")


def _oauth_config() -> dict:
    try:
        cfg = st.secrets["google_oauth"]
        required = ["client_id", "client_secret", "cookie_secret", "redirect_uri"]
        missing = [key for key in required if not cfg.get(key)]
        if missing:
            raise RuntimeError("Missing Streamlit secret(s): " + ", ".join(missing))
        return {key: str(cfg[key]) for key in required}
    except Exception as exc:
        raise RuntimeError(
            "Google OAuth is not configured. Add a [google_oauth] section in Streamlit Secrets."
        ) from exc


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _make_oauth_state(cookie_secret: str) -> str:
    payload = {"ts": int(time.time()), "nonce": secrets.token_urlsafe(18)}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(cookie_secret.encode("utf-8"), raw, hashlib.sha256).digest()
    return _b64url(raw) + "." + _b64url(sig)


def _verify_oauth_state(state: str, cookie_secret: str, max_age: int = 900) -> None:
    try:
        raw_part, sig_part = state.split(".", 1)
        raw = _b64url_decode(raw_part)
        supplied = _b64url_decode(sig_part)
        expected = hmac.new(cookie_secret.encode("utf-8"), raw, hashlib.sha256).digest()
        if not hmac.compare_digest(supplied, expected):
            raise ValueError("signature mismatch")
        payload = json.loads(raw.decode("utf-8"))
        ts = int(payload["ts"])
        if abs(int(time.time()) - ts) > max_age:
            raise ValueError("state expired")
    except Exception as exc:
        raise RuntimeError("Google login state is invalid or expired. Start the login again.") from exc


def google_login_url() -> str:
    cfg = _oauth_config()
    state = _make_oauth_state(cfg["cookie_secret"])
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "response_type": "code",
        "scope": "openid email profile " + SCOPES[0],
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "select_account",
        "state": state,
    }
    return GOOGLE_AUTH_URL + "?" + urlencode(params)


def _exchange_google_code(code: str, state: str) -> None:
    cfg = _oauth_config()
    _verify_oauth_state(state, cfg["cookie_secret"])
    response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "redirect_uri": cfg["redirect_uri"],
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    if not response.ok:
        try:
            detail = response.json().get("error_description") or response.json().get("error")
        except Exception:
            detail = response.text[:300]
        raise RuntimeError(f"Google token exchange failed: {detail}")

    token = response.json()
    access_token = token.get("access_token")
    if not access_token:
        raise RuntimeError("Google did not return an access token.")

    user_resp = requests.get(
        GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}, timeout=20
    )
    user = user_resp.json() if user_resp.ok else {}
    st.session_state["google_oauth_token"] = access_token
    st.session_state["google_oauth_email"] = user.get("email", "Google user")
    st.session_state["google_oauth_expires_at"] = int(time.time()) + int(token.get("expires_in", 3600))


def handle_google_oauth_callback() -> None:
    code = st.query_params.get("code")
    state = st.query_params.get("state")
    error = st.query_params.get("error")
    if error:
        st.query_params.clear()
        raise RuntimeError(f"Google sign-in was not completed: {error}")
    if code and state:
        _exchange_google_code(str(code), str(state))
        st.query_params.clear()
        st.rerun()


def oauth_status() -> tuple[bool, str | None]:
    token = st.session_state.get("google_oauth_token")
    expires_at = int(st.session_state.get("google_oauth_expires_at", 0) or 0)
    if token and expires_at > int(time.time()) + 30:
        return True, st.session_state.get("google_oauth_email")
    if token:
        st.session_state.pop("google_oauth_token", None)
        st.session_state.pop("google_oauth_email", None)
        st.session_state.pop("google_oauth_expires_at", None)
    return False, None


def oauth_access_token() -> str:
    token = st.session_state.get("google_oauth_token")
    if not token:
        raise RuntimeError("Google session has expired. Sign in again.")
    return str(token)


def google_logout() -> None:
    for key in ["google_oauth_token", "google_oauth_email", "google_oauth_expires_at", "oauth_df", "oauth_context"]:
        st.session_state.pop(key, None)
    st.rerun()


def connection_fetch_panel(access_token: str, session_prefix: str) -> pd.DataFrame:
    try:
        properties = list_properties(access_token)
    except Exception as exc:
        st.error(str(exc))
        return pd.DataFrame()

    if not properties:
        st.warning("No Search Console properties were returned for these credentials.")
        return pd.DataFrame()

    property_lookup = {p.get("siteUrl", ""): p.get("permissionLevel", "") for p in properties}
    site_url = st.selectbox(
        "Search Console property", list(property_lookup.keys()), key=f"{session_prefix}_site"
    )
    st.caption(f"Permission: {property_lookup.get(site_url, 'unknown')}")

    presets = ["Last 28 days", "Last 90 days", "Custom"]
    preset = st.radio("Date range", presets, horizontal=True, key=f"{session_prefix}_preset")
    default_end = date.today() - timedelta(days=3)
    if preset == "Last 90 days":
        start_date = default_end - timedelta(days=89)
        end_date = default_end
        st.caption(f"{start_date.isoformat()} → {end_date.isoformat()}")
    elif preset == "Custom":
        c1, c2 = st.columns(2)
        start_date = c1.date_input(
            "Start date", value=default_end - timedelta(days=27), key=f"{session_prefix}_start"
        )
        end_date = c2.date_input("End date", value=default_end, key=f"{session_prefix}_end")
    else:
        start_date = default_end - timedelta(days=27)
        end_date = default_end
        st.caption(f"{start_date.isoformat()} → {end_date.isoformat()}")

    search_type = st.selectbox(
        "Search type", ["web", "image", "video", "news"], key=f"{session_prefix}_type"
    )

    if st.button("Fetch Search Console data", type="primary", key=f"{session_prefix}_fetch"):
        if start_date > end_date:
            st.error("Start date must be before or equal to end date.")
        else:
            with st.spinner("Fetching query × page rows from Search Console..."):
                try:
                    fetched = fetch_gsc_data(access_token, site_url, start_date, end_date, search_type)
                except Exception as exc:
                    st.error(str(exc))
                    return pd.DataFrame()

            st.session_state[f"{session_prefix}_df"] = fetched
            st.session_state[f"{session_prefix}_context"] = {
                "site_url": site_url,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "search_type": search_type,
            }
            st.success(f"Fetched {len(fetched):,} query × page rows.")

    context = st.session_state.get(f"{session_prefix}_context", {})
    if context.get("site_url") == site_url and f"{session_prefix}_df" in st.session_state:
        df = st.session_state[f"{session_prefix}_df"]
        st.caption(
            f"Loaded {len(df):,} rows · {context.get('start_date')} → {context.get('end_date')} · "
            f"{context.get('search_type')}"
        )
        return df

    return pd.DataFrame()


def sample_csv_bytes() -> bytes:
    sample = pd.DataFrame(
        [
            {
                "query": "enterprise nas storage",
                "page": "https://example.com/enterprise-nas/",
                "clicks": 85,
                "impressions": 2420,
                "ctr": 0.0351,
                "position": 5.4,
            },
            {
                "query": "enterprise nas storage",
                "page": "https://example.com/nas-storage/",
                "clicks": 31,
                "impressions": 1180,
                "ctr": 0.0263,
                "position": 7.8,
            },
        ]
    )
    return sample.to_csv(index=False).encode("utf-8")


# ----------------------------- UI -----------------------------------------
try:
    handle_google_oauth_callback()
except Exception as exc:
    st.error(str(exc))

st.title("Keyword Cannibalization Finder")
st.caption(
    "Find queries where multiple URLs compete for meaningful Google Search visibility. "
    "Use Google sign-in, a service-account JSON key, or a query × page CSV."
)

with st.expander("What counts as a cannibalization candidate?", expanded=False):
    st.markdown(
        """
A keyword ranking through two URLs is **not automatically a problem**. This tool prioritizes overlaps where the second URL has meaningful impressions and visibility.

The score considers the competing URL's impression share, total demand, whether both URLs rank in the top 20, and how close their positions are. Always compare search intent before merging or redirecting pages.
        """
    )

source = st.radio(
    "Choose data source",
    ["Sign in with Google", "Upload service-account JSON", "Upload query × page CSV"],
    horizontal=True,
)

st.divider()
df = pd.DataFrame()

if source == "Sign in with Google":
    st.subheader("1. Connect your Google Search Console account")
    logged_in, user_email = oauth_status()

    if not logged_in:
        st.info(
            "Use the Google account that already has access to your Search Console property. "
            "The app requests read-only Search Console access."
        )
        try:
            login_url = google_login_url()
            st.link_button("Sign in with Google", login_url, type="primary")
        except Exception as exc:
            st.error(str(exc))
            st.caption("You can still use the JSON or CSV options.")
    else:
        c1, c2 = st.columns([5, 1])
        c1.success(f"Connected as {user_email or 'Google user'}")
        if c2.button("Log out"):
            google_logout()
        try:
            token = oauth_access_token()
            df = connection_fetch_panel(token, "oauth")
        except Exception as exc:
            st.error(str(exc))

elif source == "Upload service-account JSON":
    st.subheader("1. Upload a Google service-account JSON key")
    st.caption(
        "The JSON is used in memory for this session and is not written to your GitHub repository."
    )
    uploaded_json = st.file_uploader("Upload JSON key", type=["json"], key="service_json")

    if uploaded_json:
        try:
            token, service_email = service_account_token(uploaded_json)
            st.success("Service account authenticated.")
            st.code(service_email, language=None)
            st.info(
                "This service-account email must have access to the Search Console property. "
                "If you cannot add it in GSC, use 'Sign in with Google' instead."
            )
            df = connection_fetch_panel(token, "service")
        except Exception as exc:
            st.error(str(exc))

else:
    st.subheader("1. Upload query × page data")
    st.write("Required columns: `query`, `page`, `clicks`, `impressions`, `ctr`, `position`.")
    st.warning(
        "The normal Search Console export creates separate Queries.csv and Pages.csv files. "
        "Those files cannot be reliably joined for cannibalization because the query → URL relationship is missing."
    )
    st.download_button(
        "Download sample CSV format",
        data=sample_csv_bytes(),
        file_name="query_page_sample.csv",
        mime="text/csv",
    )
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="query_page_csv")
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            normalized = normalize_csv_columns(df)
            missing = REQUIRED_COLUMNS.difference(normalized.columns)
            if missing:
                st.error(
                    "This CSV is missing: " + ", ".join(sorted(missing)) + ". "
                    "You need query and page together in the same export."
                )
                df = pd.DataFrame()
            else:
                df = normalized
                st.success(f"Loaded {len(df):,} rows.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

if not df.empty:
    st.divider()
    st.subheader("2. Detection settings")
    c1, c2, c3, c4 = st.columns(4)
    min_total_impressions = c1.number_input(
        "Min total impressions/query", min_value=1, value=50, step=10
    )
    min_secondary_share_pct = c2.slider(
        "Min competing URL share", min_value=1, max_value=50, value=10, step=1
    )
    min_secondary_impressions = c3.number_input(
        "Min competing URL impressions", min_value=1, value=10, step=5
    )
    max_position = c4.number_input(
        "Max avg position to consider", min_value=1, max_value=100, value=50, step=5
    )
    exclude_regex = st.text_input(
        "Optional query exclusions (regex)", placeholder="brandname|login|support"
    )

    try:
        summary, details = detect_cannibalization(
            df,
            min_total_impressions=int(min_total_impressions),
            min_secondary_share=min_secondary_share_pct / 100,
            min_secondary_impressions=int(min_secondary_impressions),
            max_position=float(max_position),
            exclude_regex=exclude_regex,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("3. Cannibalization opportunities")

    if summary.empty:
        st.success("No candidates matched the current thresholds.")
    else:
        high_count = int((summary["severity"] == "High").sum())
        medium_count = int((summary["severity"] == "Medium").sum())
        low_count = int((summary["severity"] == "Low").sum())
        total_imp = int(summary["impressions"].sum())

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Candidates", f"{len(summary):,}")
        m2.metric("High priority", f"{high_count:,}")
        m3.metric("Medium", f"{medium_count:,}")
        m4.metric("Low", f"{low_count:,}")
        m5.metric("Affected impressions", f"{total_imp:,}")

        with st.expander("How the 0–100 score works", expanded=False):
            st.markdown(
                """
                **The score is a prioritization metric created by this tool — it is not a Google metric.**

                - Up to **45 points**: how much impression share the strongest competing URL receives.
                - Up to **20 points**: total impressions for the keyword.
                - **20 points**: both primary and competing URLs rank in the Top 20.
                - **15 points**: their average rankings are within 5 positions of each other.

                **70–100 = High**, **45–69 = Medium**, **0–44 = Low**. Always compare search intent before consolidating pages.
                """
            )

        st.markdown("#### Find an opportunity")
        f1, f2, f3, f4 = st.columns([2.2, 1.2, 1.2, 1.2])
        query_filter = f1.text_input("Search keyword", placeholder="e.g. air gapped backup")
        severity_filter = f2.multiselect(
            "Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"]
        )
        min_score_filter = f3.slider("Minimum score", 0, 100, 0)
        sort_by = f4.selectbox("Sort by", ["Score", "Impressions", "URLs", "Position gap"])

        filtered = summary.copy()
        if query_filter.strip():
            filtered = filtered[
                filtered["query"].str.contains(query_filter.strip(), case=False, na=False)
            ]
        filtered = filtered[
            filtered["severity"].isin(severity_filter) & (filtered["score"] >= min_score_filter)
        ]

        sort_map = {
            "Score": ("score", False),
            "Impressions": ("impressions", False),
            "URLs": ("urls", False),
            "Position gap": ("position_gap", True),
        }
        sort_col, ascending = sort_map[sort_by]
        filtered = filtered.sort_values(sort_col, ascending=ascending)

        if filtered.empty:
            st.warning("No opportunities match the current filters.")
        else:
            compact = filtered.copy()
            compact["Primary URL"] = compact["primary_url"].map(compact_url)
            compact["Main competing URL"] = compact["competing_url"].map(compact_url)
            compact = compact.rename(
                columns={
                    "query": "Keyword",
                    "severity": "Severity",
                    "score": "Score",
                    "urls": "URLs",
                    "impressions": "Impressions",
                    "primary_position": "Primary pos.",
                    "competing_position": "Competing pos.",
                    "position_gap": "Gap",
                }
            )
            compact_cols = [
                "Keyword", "Severity", "Score", "URLs", "Impressions",
                "Primary URL", "Primary pos.", "Main competing URL", "Competing pos.", "Gap"
            ]
            st.dataframe(
                compact[compact_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                    "Impressions": st.column_config.NumberColumn("Impressions", format="%d"),
                    "Primary pos.": st.column_config.NumberColumn("Primary pos.", format="%.2f"),
                    "Competing pos.": st.column_config.NumberColumn("Competing pos.", format="%.2f"),
                    "Gap": st.column_config.NumberColumn("Gap", format="%.2f"),
                },
                height=min(520, 44 + 36 * len(compact)),
            )

        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "Download candidate summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="keyword_cannibalization_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )
        d2.download_button(
            "Download all conflicting URLs",
            data=details.to_csv(index=False).encode("utf-8"),
            file_name="keyword_cannibalization_details.csv",
            mime="text/csv",
            use_container_width=True,
        )
        d3.download_button(
            "Download source query × page data",
            data=aggregate_query_page(df).to_csv(index=False).encode("utf-8"),
            file_name="query_page_source_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()
        st.subheader("4. Inspect a keyword conflict")
        query_options = filtered["query"].tolist() if not filtered.empty else summary["query"].tolist()
        selected_query = st.selectbox("Select keyword", query_options)
        selected_summary = summary[summary["query"] == selected_query].iloc[0]
        query_details = (
            details[details["query"] == selected_query]
            .copy()
            .sort_values(["impressions", "clicks"], ascending=False)
            .reset_index(drop=True)
        )

        score_value = int(selected_summary["score"])
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Priority score", score_value)
        a2.metric("Severity", selected_summary["severity"])
        a3.metric("Ranking URLs", int(selected_summary["urls"]))
        a4.metric("Total impressions", f"{int(selected_summary['impressions']):,}")
        a5.metric("Primary vs competitor gap", selected_summary["position_gap"])
        st.progress(score_value / 100.0)

        primary_row = query_details.iloc[0]
        competitor_row = query_details.iloc[1]
        c_primary, c_vs, c_comp = st.columns([1, 0.16, 1])
        with c_primary:
            st.markdown(
                f"""
                <div class="seo-card">
                    <h4>Primary URL</h4>
                    <div class="seo-url"><a href="{primary_row['page']}" target="_blank">{compact_url(primary_row['page'], 95)}</a></div>
                    <div class="seo-kpi">Position {float(primary_row['position']):.2f}</div>
                    <div class="seo-muted">{int(primary_row['impressions']):,} impressions · {float(primary_row['impression_share'])*100:.1f}% share · {int(primary_row['clicks']):,} clicks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_vs:
            st.markdown("<div style='text-align:center;padding-top:65px;font-weight:700;'>VS</div>", unsafe_allow_html=True)
        with c_comp:
            st.markdown(
                f"""
                <div class="seo-card">
                    <h4>Main competing URL</h4>
                    <div class="seo-url"><a href="{competitor_row['page']}" target="_blank">{compact_url(competitor_row['page'], 95)}</a></div>
                    <div class="seo-kpi">Position {float(competitor_row['position']):.2f}</div>
                    <div class="seo-muted">{int(competitor_row['impressions']):,} impressions · {float(competitor_row['impression_share'])*100:.1f}% share · {int(competitor_row['clicks']):,} clicks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Recommended next step")
        st.info(selected_summary["recommendation"])

        breakdown = score_breakdown(
            float(selected_summary["impressions"]),
            float(selected_summary["competing_impression_share"]),
            float(selected_summary["primary_position"]),
            float(selected_summary["competing_position"]),
        )
        with st.expander("Why this keyword received this score", expanded=False):
            b1, b2, b3, b4 = st.columns(4)
            values = list(breakdown.items())
            for col, (label, value) in zip([b1, b2, b3, b4], values):
                max_points = {"Competing URL share": 45, "Keyword demand": 20, "Both URLs in Top 20": 20, "Rankings within 5 positions": 15}[label]
                col.metric(label, f"{value:g}/{max_points}")

        st.markdown("#### All URLs ranking for this keyword")
        all_urls = query_details.copy()
        all_urls.insert(0, "Role", ["Primary"] + ["Competitor"] * (len(all_urls) - 1))
        all_urls["URL"] = all_urls["page"].map(compact_url)
        all_urls["CTR"] = (all_urls["ctr"] * 100).round(2)
        all_urls["Impression share"] = (all_urls["impression_share"] * 100).round(1)
        all_urls = all_urls.rename(
            columns={"clicks": "Clicks", "impressions": "Impressions", "position": "Position"}
        )
        st.dataframe(
            all_urls[["Role", "URL", "Clicks", "Impressions", "CTR", "Position", "Impression share"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "CTR": st.column_config.NumberColumn("CTR", format="%.2f%%"),
                "Position": st.column_config.NumberColumn("Position", format="%.2f"),
                "Impression share": st.column_config.ProgressColumn(
                    "Impression share", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

        if len(query_details) > 2:
            with st.expander(f"Show {len(query_details)-2} additional competing URLs", expanded=False):
                for idx, row in query_details.iloc[2:].iterrows():
                    st.markdown(
                        f"**{idx+1}. [{compact_url(row['page'], 100)}]({row['page']})**  · "
                        f"Position **{float(row['position']):.2f}** · "
                        f"{int(row['impressions']):,} impressions · "
                        f"{float(row['impression_share'])*100:.1f}% share"
                    )

        st.markdown("#### Impression split across ranking URLs")
        chart_df = query_details.copy()
        chart_df["URL"] = chart_df["page"].map(lambda x: compact_url(x, 55))
        chart_df = chart_df.set_index("URL")[["impressions"]]
        st.bar_chart(chart_df)

st.divider()
st.caption(
    "v5 · Improved conflict-first UI · Google OAuth + service-account JSON + CSV · query × page overlap scoring. "
    "Planned next: daily URL-switching detection and page-intent similarity."
)
