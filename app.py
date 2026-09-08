import base64
import hashlib
import hmac
import io
import json
import re
import secrets
import time
from datetime import date, timedelta
from difflib import SequenceMatcher
from urllib.parse import quote, urlencode

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
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


def risk_score(total_impressions: float, second_share: float, pos1: float, pos2: float, url_count: int) -> int:
    """Internal heuristic used to order issues. It is not a Google metric."""
    score = 0.0
    score += 40 * min(second_share / 0.40, 1.0)
    score += 20 * min(total_impressions / 1000.0, 1.0)
    if pos1 <= 20 and pos2 <= 20:
        score += 20
    if abs(pos1 - pos2) <= 5:
        score += 15
    if url_count >= 3:
        score += 5
    return int(round(min(score, 100)))


def risk_label(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def classify_overlap(primary_pos: float, secondary_pos: float, primary_share: float, secondary_share: float) -> str:
    gap = abs(primary_pos - secondary_pos)
    both_top20 = primary_pos <= 20 and secondary_pos <= 20
    if secondary_share >= 0.25 and both_top20 and gap <= 5:
        return "Likely Cannibalization"
    if secondary_share >= 0.15 and (both_top20 or gap <= 7):
        return "Possible Cannibalization"
    if secondary_share >= 0.08:
        return "Minor Overlap"
    return "No Action Needed"


def confidence_label(status: str, total_impressions: float, secondary_impressions: float, secondary_share: float) -> str:
    if status == "Likely Cannibalization" and total_impressions >= 100 and secondary_impressions >= 25 and secondary_share >= 0.25:
        return "Strong"
    if status in {"Likely Cannibalization", "Possible Cannibalization"} and total_impressions >= 50:
        return "Moderate"
    if status == "No Action Needed" and secondary_share < 0.05:
        return "Strong"
    return "Weak"


def evidence_text(primary_pos: float, secondary_pos: float, primary_share: float, secondary_share: float, url_count: int) -> str:
    parts = [f"{url_count} ranking URLs", f"secondary URL has {secondary_share:.0%} of impressions"]
    if primary_pos <= 20 and secondary_pos <= 20:
        parts.append("both main URLs rank in Top 20")
    gap = abs(primary_pos - secondary_pos)
    if gap <= 5:
        parts.append(f"positions are only {gap:.1f} apart")
    if primary_share >= 0.75:
        parts.append(f"primary URL dominates at {primary_share:.0%}")
    return " • ".join(parts)


def action_type(status: str) -> str:
    return {
        "Likely Cannibalization": "Review for consolidation",
        "Possible Cannibalization": "Retarget / differentiate",
        "Minor Overlap": "Monitor",
        "No Action Needed": "No action",
    }.get(status, "Review")


def base_action(status: str, primary_pos: float, secondary_pos: float, secondary_share: float) -> str:
    if status == "Likely Cannibalization":
        return "Review intent. If the pages satisfy the same intent, consolidate; if not, retarget the weaker page and clarify internal links."
    if status == "Possible Cannibalization":
        return "Compare intent, titles/H1s and internal anchors. Retarget the weaker URL if the overlap is accidental."
    if status == "Minor Overlap":
        return "Monitor. No major change unless URL switching or very high page-intent similarity is detected."
    return "No action recommended. One URL appears to dominate the query; keep monitoring."


def detect_cannibalization_qp(
    qp: pd.DataFrame,
    min_total_impressions: int = 50,
    min_secondary_impressions: int = 5,
    max_position: float = 100.0,
    exclude_regex: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qp = qp.copy()
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
        if float(secondary["impressions"]) < min_secondary_impressions:
            continue

        primary_share = float(primary["impressions"] / total_imp) if total_imp else 0.0
        secondary_share = float(secondary["impressions"] / total_imp) if total_imp else 0.0
        primary_pos = float(primary["position"])
        secondary_pos = float(secondary["position"])
        url_count = int(group["page"].nunique())
        score = risk_score(total_imp, secondary_share, primary_pos, secondary_pos, url_count)
        status = classify_overlap(primary_pos, secondary_pos, primary_share, secondary_share)
        confidence = confidence_label(status, total_imp, float(secondary["impressions"]), secondary_share)

        summary_rows.append(
            {
                "query": query,
                "status": status,
                "risk": risk_label(score),
                "confidence": confidence,
                "risk_score": score,
                "ranking_urls": url_count,
                "clicks": round(total_clicks, 0),
                "impressions": round(total_imp, 0),
                "primary_url": primary["page"],
                "primary_position": round(primary_pos, 2),
                "primary_impression_share": round(primary_share, 4),
                "competing_url": secondary["page"],
                "competing_position": round(secondary_pos, 2),
                "competing_impression_share": round(secondary_share, 4),
                "position_gap": round(abs(primary_pos - secondary_pos), 2),
                "why_flagged": evidence_text(primary_pos, secondary_pos, primary_share, secondary_share, url_count),
                "action": action_type(status),
                "recommended_action": base_action(status, primary_pos, secondary_pos, secondary_share),
            }
        )

        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            detail_rows.append(
                {
                    "query": query,
                    "status": status,
                    "risk": risk_label(score),
                    "role": "Primary" if rank == 1 else f"Competitor #{rank-1}",
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
        status_order = pd.Categorical(
            summary["status"],
            categories=["Likely Cannibalization", "Possible Cannibalization", "Minor Overlap", "No Action Needed"],
            ordered=True,
        )
        summary = (
            summary.assign(_status=status_order)
            .sort_values(["_status", "risk_score", "impressions"], ascending=[True, False, False])
            .drop(columns="_status")
            .reset_index(drop=True)
        )
    return summary, details


def detect_cannibalization(
    df: pd.DataFrame,
    min_total_impressions: int = 50,
    min_secondary_impressions: int = 5,
    max_position: float = 100.0,
    exclude_regex: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return detect_cannibalization_qp(
        aggregate_query_page(df), min_total_impressions, min_secondary_impressions, max_position, exclude_regex
    )


def _simple_stem(token: str) -> str:
    token = re.sub(r"[^a-z0-9]", "", token.lower())
    if len(token) > 5 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 5 and token.endswith("ed"):
        token = token[:-2]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
    elif len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def _variant_key(query: str) -> str:
    query = re.sub(r"[-_/]+", " ", normalize_query(query))
    tokens = [_simple_stem(t) for t in re.findall(r"[a-z0-9]+", query)]
    return "".join(t for t in tokens if t)


def build_query_families(qp: pd.DataFrame, similarity_threshold: float = 0.86) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Groups close spelling/plural/hyphen variants; deliberately conservative, not semantic clustering."""
    q_imp = qp.groupby("query", as_index=False)["impressions"].sum().sort_values("impressions", ascending=False)
    buckets: dict[str, list[tuple[str, str]]] = {}
    mapping: dict[str, str] = {}
    members: dict[str, list[str]] = {}

    for row in q_imp.itertuples(index=False):
        query = str(row.query)
        key = _variant_key(query)
        bucket_key = key[:4]
        chosen = None
        for rep_query, rep_key in buckets.get(bucket_key, []):
            ratio = SequenceMatcher(None, key, rep_key).ratio()
            contained = (key in rep_key or rep_key in key) and abs(len(key) - len(rep_key)) <= 4
            if ratio >= similarity_threshold or contained:
                chosen = rep_query
                break
        if chosen is None:
            chosen = query
            buckets.setdefault(bucket_key, []).append((query, key))
            members[chosen] = []
        mapping[query] = chosen
        members.setdefault(chosen, []).append(query)

    work = qp.copy()
    work["family"] = work["query"].map(mapping)
    rows = []
    for (family, page), group in work.groupby(["family", "page"]):
        impressions = float(group["impressions"].sum())
        clicks = float(group["clicks"].sum())
        rows.append({
            "query": family,
            "page": page,
            "clicks": clicks,
            "impressions": impressions,
            "ctr": clicks / impressions if impressions else 0.0,
            "position": weighted_position(group),
        })
    family_qp = pd.DataFrame(rows)
    member_rows = []
    for family, qs in members.items():
        family_imp = float(q_imp[q_imp["query"].isin(qs)]["impressions"].sum())
        member_rows.append({
            "family": family,
            "variant_count": len(qs),
            "variants": " | ".join(qs[:12]) + (" | …" if len(qs) > 12 else ""),
            "family_impressions": round(family_imp, 0),
        })
    return family_qp, pd.DataFrame(member_rows)


STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","how","in","is","it","of","on","or","that","the","this","to","what","when","where","which","with","your","you"
}


def _text_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
    return {_simple_stem(w) for w in words if w not in STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_page_profile(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SEO-Cannibalization-Audit/1.0)"}
    response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    h1 = soup.find("h1")
    h1_text = h1.get_text(" ", strip=True) if h1 else ""
    meta = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    desc = meta.get("content", "").strip() if meta else ""
    body = " ".join(soup.stripped_strings)[:50000]
    return {"url": response.url, "title": title, "h1": h1_text, "description": desc, "body": body}


def page_intent_similarity(url_a: str, url_b: str) -> tuple[int, dict, dict, dict]:
    a = fetch_page_profile(url_a)
    b = fetch_page_profile(url_b)
    title_sim = _jaccard(_text_tokens(a["title"]), _text_tokens(b["title"]))
    h1_sim = _jaccard(_text_tokens(a["h1"]), _text_tokens(b["h1"]))
    desc_sim = _jaccard(_text_tokens(a["description"]), _text_tokens(b["description"]))
    body_sim = _jaccard(_text_tokens(a["body"]), _text_tokens(b["body"]))
    score = int(round(100 * (0.35 * title_sim + 0.25 * h1_sim + 0.10 * desc_sim + 0.30 * body_sim)))
    components = {
        "Title similarity": int(round(title_sim * 100)),
        "H1 similarity": int(round(h1_sim * 100)),
        "Meta similarity": int(round(desc_sim * 100)),
        "Body-topic overlap": int(round(body_sim * 100)),
    }
    return score, components, a, b


def intent_interpretation(score: int, status: str) -> tuple[str, str]:
    if score >= 65 and status in {"Likely Cannibalization", "Possible Cannibalization"}:
        return "High intent overlap", "Strong merge/consolidation candidate. Manually verify unique value before redirecting anything."
    if score >= 40:
        return "Moderate intent overlap", "Keep both only if their purposes are distinct; otherwise retarget the weaker page and clarify internal linking."
    return "Low intent overlap", "Likely different intents. Prefer differentiation and internal-link cleanup over merging."

def format_summary_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if out.empty:
        return out
    for col in ["primary_impression_share", "competing_impression_share"]:
        if col in out.columns:
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



def fetch_gsc_daily_query(
    access_token: str,
    site_url: str,
    query: str,
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
            "dimensions": ["date", "page"],
            "dimensionFilterGroups": [{"filters": [{"dimension": "query", "operator": "equals", "expression": query}]}],
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
            all_rows.append({
                "date": keys[0] if len(keys) > 0 else "",
                "page": keys[1] if len(keys) > 1 else "",
                "clicks": row.get("clicks", 0),
                "impressions": row.get("impressions", 0),
                "ctr": row.get("ctr", 0),
                "position": row.get("position", 0),
            })
        if len(rows) < row_limit:
            break
        start_row += row_limit
    return pd.DataFrame(all_rows)


def url_switching_summary(daily: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if daily.empty:
        return {"switches": 0, "winning_urls": 0, "signal": "No data"}, pd.DataFrame()
    winners = (
        daily.sort_values(["date", "impressions", "clicks"], ascending=[True, False, False])
        .groupby("date", as_index=False)
        .first()
        .sort_values("date")
        .reset_index(drop=True)
    )
    prev = None
    switches = 0
    for page in winners["page"]:
        if prev is not None and page != prev:
            switches += 1
        prev = page
    distinct = int(winners["page"].nunique())
    if switches >= 5 and distinct >= 2:
        signal = "Strong switching signal"
    elif switches >= 2 and distinct >= 2:
        signal = "Moderate switching signal"
    elif distinct >= 2:
        signal = "Weak switching signal"
    else:
        signal = "Stable primary URL"
    return {"switches": switches, "winning_urls": distinct, "signal": signal, "days": len(winners)}, winners

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

    st.session_state[f"{session_prefix}_access_token"] = access_token
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
    "Find meaningful URL conflicts in Google Search Console, separate real cannibalization from harmless overlap, "
    "and get evidence before making SEO changes."
)

with st.expander("How this tool decides whether an overlap is a real problem", expanded=False):
    st.markdown(
        """
**Multiple URLs ranking for one query is not automatically cannibalization.** The tool first finds URL overlap, then evaluates:

- how much impression share the second URL receives,
- whether both main URLs rank in the Top 20,
- how close their positions are,
- how many URLs are involved,
- and, on demand, whether Google switches the winning URL over time and whether the pages appear to target the same intent.

The internal **risk score** is used only for sorting; it is **not a Google metric**. The main output is the evidence-based status: **Likely Cannibalization, Possible Cannibalization, Minor Overlap, or No Action Needed**.
        """
    )

source = st.radio(
    "Choose data source",
    ["Sign in with Google", "Upload service-account JSON", "Upload query × page CSV"],
    horizontal=True,
)

st.divider()
df = pd.DataFrame()
active_session_prefix = None

if source == "Sign in with Google":
    st.subheader("1. Connect your Google Search Console account")
    logged_in, user_email = oauth_status()
    if not logged_in:
        st.info("Use the Google account that already has access to your Search Console property. Read-only GSC access is requested.")
        try:
            st.link_button("Sign in with Google", google_login_url(), type="primary")
        except Exception as exc:
            st.error(str(exc))
            st.caption("You can still use the JSON or CSV options.")
    else:
        c1, c2 = st.columns([5, 1])
        c1.success(f"Connected as {user_email or 'Google user'}")
        if c2.button("Log out"):
            google_logout()
        try:
            df = connection_fetch_panel(oauth_access_token(), "oauth")
            active_session_prefix = "oauth"
        except Exception as exc:
            st.error(str(exc))

elif source == "Upload service-account JSON":
    st.subheader("1. Upload a Google service-account JSON key")
    st.caption("The JSON is used in memory for this session and is not written to GitHub.")
    uploaded_json = st.file_uploader("Upload JSON key", type=["json"], key="service_json")
    if uploaded_json:
        try:
            token, service_email = service_account_token(uploaded_json)
            st.success("Service account authenticated.")
            st.code(service_email, language=None)
            st.info("This service-account email must have Search Console property access.")
            df = connection_fetch_panel(token, "service")
            active_session_prefix = "service"
        except Exception as exc:
            st.error(str(exc))

else:
    st.subheader("1. Upload query × page data")
    st.write("Required columns: `query`, `page`, `clicks`, `impressions`, `ctr`, `position`.")
    st.warning(
        "Separate Queries.csv and Pages.csv exports cannot be reliably joined because the query → URL relationship is missing."
    )
    st.download_button("Download sample CSV format", data=sample_csv_bytes(), file_name="query_page_sample.csv", mime="text/csv")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="query_page_csv")
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            normalized = normalize_csv_columns(df)
            missing = REQUIRED_COLUMNS.difference(normalized.columns)
            if missing:
                st.error("This CSV is missing: " + ", ".join(sorted(missing)) + ". Query and page must be in the same rows.")
                df = pd.DataFrame()
            else:
                df = normalized
                st.success(f"Loaded {len(df):,} rows.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

if not df.empty:
    st.divider()
    st.subheader("2. Detection settings")
    c1, c2, c3 = st.columns(3)
    min_total_impressions = c1.number_input("Min total impressions/query", min_value=1, value=50, step=10)
    min_secondary_impressions = c2.number_input("Min secondary URL impressions", min_value=1, value=5, step=5)
    max_position = c3.number_input("Max avg position to consider", min_value=1, max_value=100, value=50, step=5)
    exclude_regex = st.text_input("Optional query exclusions (regex)", placeholder="brandname|login|support")

    try:
        qp = aggregate_query_page(df)
        summary, details = detect_cannibalization_qp(
            qp,
            min_total_impressions=int(min_total_impressions),
            min_secondary_impressions=int(min_secondary_impressions),
            max_position=float(max_position),
            exclude_regex=exclude_regex,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("3. Cannibalization audit")
    if summary.empty:
        st.success("No multi-URL overlaps matched the current thresholds.")
    else:
        likely = int((summary["status"] == "Likely Cannibalization").sum())
        possible = int((summary["status"] == "Possible Cannibalization").sum())
        minor = int((summary["status"] == "Minor Overlap").sum())
        no_action = int((summary["status"] == "No Action Needed").sum())
        action_queries = set(summary.loc[summary["status"].isin(["Likely Cannibalization", "Possible Cannibalization"]), "query"])
        action_urls = set(details.loc[details["query"].isin(action_queries), "page"]) if not details.empty else set()

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Likely", f"{likely:,}")
        m2.metric("Possible", f"{possible:,}")
        m3.metric("Minor overlap", f"{minor:,}")
        m4.metric("No action", f"{no_action:,}")
        m5.metric("URLs to review", f"{len(action_urls):,}")

        tab_queries, tab_families = st.tabs(["Exact-query conflicts", "Keyword-family view"])

        with tab_queries:
            f1, f2, f3 = st.columns([2, 2, 1])
            query_filter = f1.text_input("Filter queries", placeholder="air gapped backup", key="exact_filter")
            status_options = ["Likely Cannibalization", "Possible Cannibalization", "Minor Overlap", "No Action Needed"]
            status_filter = f2.multiselect(
                "Status",
                status_options,
                default=["Likely Cannibalization", "Possible Cannibalization"],
                key="status_filter",
            )
            risk_filter = f3.multiselect("Risk", ["High", "Medium", "Low"], default=["High", "Medium", "Low"], key="risk_filter")

            filtered = summary.copy()
            if query_filter.strip():
                filtered = filtered[filtered["query"].str.contains(query_filter.strip(), case=False, na=False)]
            filtered = filtered[filtered["status"].isin(status_filter) & filtered["risk"].isin(risk_filter)]

            display_cols = [
                "query", "status", "risk", "confidence", "ranking_urls", "impressions",
                "primary_url", "competing_url", "primary_position", "competing_position",
                "position_gap", "why_flagged", "action", "recommended_action"
            ]
            st.dataframe(
                format_summary_for_display(filtered)[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "query": st.column_config.TextColumn("Query"),
                    "status": st.column_config.TextColumn("Assessment", help="Evidence-based classification; multiple URLs alone are not considered cannibalization."),
                    "risk": st.column_config.TextColumn("Risk", help="High/Medium/Low priority derived from visibility split, demand and ranking proximity."),
                    "confidence": st.column_config.TextColumn("Confidence", help="Strength of the evidence supporting the assessment."),
                    "ranking_urls": st.column_config.NumberColumn("Ranking URLs", help="Unique URLs from your site that received impressions for this query."),
                    "why_flagged": st.column_config.TextColumn("Why flagged"),
                    "action": st.column_config.TextColumn("Action"),
                    "recommended_action": st.column_config.TextColumn("Recommended next step"),
                },
            )

        with tab_families:
            st.caption("Groups close spelling/plural/hyphen variants (for example, air-gapped / air gapped / airgapped). This is conservative lexical grouping, not semantic keyword clustering.")
            try:
                family_qp, family_members = build_query_families(qp)
                family_summary, family_details = detect_cannibalization_qp(
                    family_qp,
                    min_total_impressions=int(min_total_impressions),
                    min_secondary_impressions=int(min_secondary_impressions),
                    max_position=float(max_position),
                    exclude_regex=exclude_regex,
                )
                family_summary = family_summary.merge(family_members, left_on="query", right_on="family", how="left")
                family_summary = family_summary[family_summary["variant_count"].fillna(1) > 1]
                if family_summary.empty:
                    st.info("No multi-query keyword families with URL overlap were found under the current thresholds.")
                else:
                    fam_cols = ["query", "variant_count", "variants", "status", "risk", "confidence", "ranking_urls", "impressions", "primary_url", "competing_url", "why_flagged", "action"]
                    st.dataframe(
                        family_summary[fam_cols].rename(columns={"query": "family"}),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "family": st.column_config.TextColumn("Keyword family"),
                            "variant_count": st.column_config.NumberColumn("Variants"),
                            "ranking_urls": st.column_config.NumberColumn("Ranking URLs"),
                        },
                    )
            except Exception as exc:
                st.warning(f"Keyword-family grouping could not be built: {exc}")

        st.divider()
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download audit summary", data=summary.to_csv(index=False).encode("utf-8"), file_name="keyword_cannibalization_audit.csv", mime="text/csv")
        d2.download_button("Download URL-level details", data=details.to_csv(index=False).encode("utf-8"), file_name="keyword_cannibalization_url_details.csv", mime="text/csv")
        d3.download_button("Download source query × page data", data=qp.to_csv(index=False).encode("utf-8"), file_name="query_page_source_data.csv", mime="text/csv")

        st.subheader("4. Investigate one query")
        review_pool = summary[summary["status"].isin(["Likely Cannibalization", "Possible Cannibalization", "Minor Overlap"])]
        if review_pool.empty:
            review_pool = summary
        selected_query = st.selectbox("Query", review_pool["query"].tolist())
        selected_summary = summary[summary["query"] == selected_query].iloc[0]
        query_details = details[details["query"] == selected_query].copy().sort_values("impressions", ascending=False)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Assessment", selected_summary["status"])
        a2.metric("Risk", selected_summary["risk"])
        a3.metric("Confidence", selected_summary["confidence"])
        a4.metric("Ranking URLs", int(selected_summary["ranking_urls"]))
        st.info("Why flagged: " + selected_summary["why_flagged"])
        st.success("Recommended next step: " + selected_summary["recommended_action"])

        display_details = query_details.copy()
        display_details["ctr"] = (display_details["ctr"] * 100).round(2).astype(str) + "%"
        display_details["impression_share"] = (display_details["impression_share"] * 100).round(1).astype(str) + "%"
        st.markdown("#### All URLs ranking for this query")
        st.dataframe(
            display_details[["role", "page", "clicks", "impressions", "ctr", "position", "impression_share"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "role": st.column_config.TextColumn("Role"),
                "page": st.column_config.LinkColumn("URL", display_text="Open page"),
                "impression_share": st.column_config.TextColumn("Impression share"),
            },
        )

        top_two = query_details.head(2)
        if len(top_two) >= 2:
            primary_url = str(top_two.iloc[0]["page"])
            competing_url = str(top_two.iloc[1]["page"])
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Primary URL**")
                st.write(primary_url)
                st.caption(f"Position {top_two.iloc[0]['position']} · Share {top_two.iloc[0]['impression_share']:.1%}")
            with col_right:
                st.markdown("**Main competing URL**")
                st.write(competing_url)
                st.caption(f"Position {top_two.iloc[1]['position']} · Share {top_two.iloc[1]['impression_share']:.1%}")

            st.markdown("#### Page-intent check")
            st.caption("Optional on-page comparison using title, H1, meta description and visible page text. It is a heuristic, not a substitute for manual SERP-intent review.")
            if st.button("Analyze intent similarity", key=f"intent_{selected_query}"):
                with st.spinner("Reading both pages and comparing intent signals..."):
                    try:
                        sim, components, profile_a, profile_b = page_intent_similarity(primary_url, competing_url)
                        label, intent_action = intent_interpretation(sim, selected_summary["status"])
                        i1, i2 = st.columns([1, 3])
                        i1.metric("Intent similarity", f"{sim}%")
                        i2.info(f"{label}. {intent_action}")
                        st.write(components)
                        p1, p2 = st.columns(2)
                        with p1:
                            st.markdown("**Primary page signals**")
                            st.write("Title:", profile_a["title"] or "—")
                            st.write("H1:", profile_a["h1"] or "—")
                        with p2:
                            st.markdown("**Competing page signals**")
                            st.write("Title:", profile_b["title"] or "—")
                            st.write("H1:", profile_b["h1"] or "—")
                    except Exception as exc:
                        st.warning(f"Could not analyze both pages: {exc}")

        st.markdown("#### URL-switching check")
        if active_session_prefix and st.session_state.get(f"{active_session_prefix}_context"):
            ctx = st.session_state[f"{active_session_prefix}_context"]
            token = st.session_state.get(f"{active_session_prefix}_access_token")
            if st.button("Check daily URL switching", key=f"switch_{selected_query}"):
                with st.spinner("Checking which URL won each day in Search Console..."):
                    try:
                        daily = fetch_gsc_daily_query(
                            token,
                            ctx["site_url"],
                            selected_query,
                            date.fromisoformat(ctx["start_date"]),
                            date.fromisoformat(ctx["end_date"]),
                            ctx["search_type"],
                        )
                        switch_info, winners = url_switching_summary(daily)
                        s1, s2, s3 = st.columns(3)
                        s1.metric("URL switches", switch_info.get("switches", 0))
                        s2.metric("Distinct daily winners", switch_info.get("winning_urls", 0))
                        s3.metric("Switching signal", switch_info.get("signal", "No data"))
                        if not winners.empty:
                            st.caption("Daily winner = URL with the most impressions for the query on that day.")
                            st.dataframe(winners[["date", "page", "impressions", "clicks", "position"]], use_container_width=True, hide_index=True)
                    except Exception as exc:
                        st.warning(f"Could not fetch daily switching data: {exc}")
        else:
            st.caption("URL switching requires a live Search Console connection. CSV mode can still use the overlap and intent checks.")

st.divider()
st.caption(
    "v6 · Evidence-first cannibalization audit · exact-query overlap + keyword families + on-page intent similarity + on-demand URL switching."
)
