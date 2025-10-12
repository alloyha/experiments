#!/usr/bin/env python3
"""
Composio inspector — safer version with dry-run and summary options.

Usage examples:
  # quick sample (server-side limit)
  python composio_inspector.py --mode actions --app GITHUB --limit 10 --per-page 10 --max-pages 1 --output sample.json

  # dry-run: don't write large JSON, just print counts & per-app summary
  python composio_inspector.py --mode actions --app GITHUB --dry-run --limit 50

  # summary-only JSON (small)
  python composio_inspector.py --mode actions --app GITHUB --summary-only --output actions_summary.json --limit 100

  # CI: fail if any action requires auth but no connected account exists
  python composio_inspector.py --mode actions --fail-on-missing-auth --limit 50
"""
from __future__ import annotations
import os
import sys
import json
import csv
import logging
import argparse
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

# ---- Config ----
API_KEY = os.getenv("COMPOSIO_API_KEY")
BASE_V2 = os.getenv("COMPOSIO_BASE_V2", "https://backend.composio.dev/api/v2")
BASE_V3 = os.getenv("COMPOSIO_BASE_V3", "https://backend.composio.dev/api/v3")

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("composio_inspector")

if not API_KEY:
    log.error("COMPOSIO_API_KEY not set. Exiting.")
    sys.exit(1)

HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}


def build_session(retries: int = 3, backoff: float = 0.3, status_forcelist=(429, 500, 502, 503, 504)) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(HEADERS)
    return s


SESSION = build_session()


# ---- Helpers & Normalization ----
def safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        log.error("Response is not JSON (status=%s): %s", resp.status_code, resp.text[:500])
        raise


def normalize_app_key(item: Dict[str, Any]) -> Optional[str]:
    return (
        item.get("appKey")
        or item.get("app_key")
        or item.get("appName")
        or item.get("key")
        or item.get("appUniqueId")
        or item.get("uniqueId")
    )


def normalize_action_name(item: Dict[str, Any]) -> Optional[str]:
    return item.get("name") or item.get("actionName") or item.get("slug") or item.get("id")


def paginate_get(path: str, params: Optional[Dict[str, Any]] = None, per_page: int = 50, max_pages: int = 3) -> List[Dict[str, Any]]:
    """
    Defensive pagination: logs progress and enforces client max_pages/per_page.
    Default max_pages is small to avoid accidental huge downloads.
    """
    out: List[Dict[str, Any]] = []
    page = 1
    while True:
        p = dict(params or {})
        p.update({"page": page, "per_page": per_page})
        log.debug("paginate_get: requesting %s page=%d per_page=%d", path, page, per_page)
        try:
            r = SESSION.get(path, params=p, timeout=30)
            r.raise_for_status()
            data = safe_json(r)
        except requests.RequestException as exc:
            log.error("Request failed for %s page=%d: %s", path, page, exc)
            break

        if isinstance(data, dict) and "items" in data:
            items = data.get("items") or []
        elif isinstance(data, dict) and "results" in data:
            items = data.get("results") or []
        elif isinstance(data, list):
            items = data
        else:
            items = [data] if data else []

        log.info("page=%d -> got %d items", page, len(items))
        out.extend(items)

        if len(items) > 5000:
            log.warning("single page returned %d items; stopping to avoid OOM", len(items))
            break

        total = None
        if isinstance(data, dict):
            total = data.get("total") or data.get("count") or (data.get("meta") or {}).get("total")
        if total is not None:
            if len(out) >= int(total):
                break
            page += 1
            if page > max_pages:
                log.warning("reached max_pages (%d) for %s, stopping", max_pages, path)
                break
        else:
            if len(items) < per_page:
                break
            page += 1
            if page > max_pages:
                log.warning("reached max_pages (%d) for %s, stopping", max_pages, path)
                break

    return out


# ---- Inspector class ----
class ComposioInspector:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or SESSION

    # ---- Listing methods ----
    def list_apps(self, per_page: int = 50, max_pages: int = 3) -> List[Dict[str, Any]]:
        url = f"{BASE_V2.rstrip('/')}/apps"
        return paginate_get(url, per_page=per_page, max_pages=max_pages)

    def get_app(self, app_key: str) -> Optional[Dict[str, Any]]:
        url = f"{BASE_V2.rstrip('/')}/apps/{app_key}"
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            return safe_json(r)
        except requests.RequestException as e:
            log.error("Error fetching app %s: %s", app_key, e)
            return None

    def list_actions(self, apps: Optional[str] = None, use_case: Optional[str] = None, tags: Optional[str] = None, limit: Optional[int] = None, per_page: int = 50, max_pages: int = 3) -> List[Dict[str, Any]]:
        url = f"{BASE_V2.rstrip('/')}/actions"
        params = {}
        if apps:
            params["apps"] = apps
        if use_case:
            params["useCase"] = use_case
        if tags:
            params["tags"] = tags
        if limit is not None:
            params["limit"] = limit
        return paginate_get(url, params=params, per_page=per_page, max_pages=max_pages)

    def get_action(self, action_name: str) -> Optional[Dict[str, Any]]:
        url = f"{BASE_V2.rstrip('/')}/actions/{action_name}"
        try:
            r = self.session.get(url, timeout=12)
            r.raise_for_status()
            return safe_json(r)
        except requests.RequestException as e:
            log.error("Error getting action %s: %s", action_name, e)
            return None

    def list_connected_accounts(self, user_uuid: Optional[str] = None, per_page: int = 50, max_pages: int = 3) -> List[Dict[str, Any]]:
        url = f"{BASE_V2.rstrip('/')}/connectedAccounts"
        params = {}
        if user_uuid:
            params["user_uuid"] = user_uuid
        return paginate_get(url, params=params, per_page=per_page, max_pages=max_pages)

    def list_integrations(self, per_page: int = 50, max_pages: int = 3) -> List[Dict[str, Any]]:
        """
        Try v2 /integrations first; if it returns 410 Gone, try v3 /auth_configs as a fallback.
        """
        url_v2 = f"{BASE_V2.rstrip('/')}/integrations"
        try:
            r = SESSION.get(url_v2, timeout=20)
            if r.status_code == 410:
                log.warning("integrations v2 returned 410; trying v3/auth_configs fallback")
                url_v3 = f"{BASE_V3.rstrip('/')}/auth_configs"
                return paginate_get(url_v3, per_page=per_page, max_pages=max_pages)
            r.raise_for_status()
            data = safe_json(r)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "items" in data:
                return data.get("items", [])
            return paginate_get(url_v2, per_page=per_page, max_pages=max_pages)
        except requests.RequestException as e:
            log.error("list_integrations failed: %s", e)
            try:
                url_v3 = f"{BASE_V3.rstrip('/')}/auth_configs"
                return paginate_get(url_v3, per_page=per_page, max_pages=max_pages)
            except Exception as e2:
                log.error("v3 fallback also failed: %s", e2)
                return []

    # ---- Catalog builders ----
    def build_actions_catalog(self, user_uuid: Optional[str] = None, app_filter: Optional[str] = None, detailed: bool = False, workers: int = 4, per_page: int = 50, max_pages: int = 3, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        log.info("Fetching actions (server-side request)...")
        actions = self.list_actions(apps=app_filter, limit=limit, per_page=per_page, max_pages=max_pages)
        log.info("Fetched %d raw actions from server", len(actions))

        # Client-side tolerant filtering by app_filter (substring match on common fields)
        if app_filter:
            af = app_filter.lower()
            def matches_app(a: Dict[str, Any]) -> bool:
                candidates = [
                    a.get("appKey") or a.get("app_key"),
                    a.get("appName") or a.get("app_name"),
                    a.get("app") or a.get("app_id"),
                    a.get("appUniqueId")
                ]
                for cand in candidates:
                    if cand and af in str(cand).lower():
                        return True
                # also try action-level display fields
                for cand2 in (a.get("displayName") or a.get("title") or a.get("description") or a.get("tags") or []):
                    if cand2 and af in str(cand2).lower():
                        return True
                return False
            filtered = [a for a in actions if matches_app(a)]
            log.info("Client-side filtered actions: %d -> %d using app_filter='%s'", len(actions), len(filtered), app_filter)
            actions = filtered

        log.info("Fetching integrations...")
        integrations = self.list_integrations(per_page=per_page, max_pages=max_pages)

        log.info("Fetching connected accounts...")
        connected = self.list_connected_accounts(user_uuid, per_page=per_page, max_pages=max_pages) if user_uuid else []

        # Map app key -> integrations
        int_by_app: Dict[str, List[Dict[str, Any]]] = {}
        for inte in integrations:
            k = normalize_app_key(inte)
            if k:
                int_by_app.setdefault(k, []).append(inte)

        # Map app key -> connected accounts
        conn_by_app: Dict[str, List[Dict[str, Any]]] = {}
        for c in connected:
            k = normalize_app_key(c)
            if k:
                conn_by_app.setdefault(k, []).append(c)

        catalog: List[Dict[str, Any]] = []
        if detailed:
            log.info("Detailed mode: fetching individual action details with %d workers", workers)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {}
                for action in actions:
                    name = normalize_action_name(action)
                    if name:
                        futures[ex.submit(self.get_action, name)] = action
                for fut in as_completed(futures):
                    action = futures[fut]
                    detail = None
                    try:
                        detail = fut.result()
                    except Exception as e:
                        log.error("Failed fetching detail for action: %s", e)
                    catalog.append(self._action_entry(action, detail, int_by_app, conn_by_app, user_uuid))
        else:
            for action in actions:
                catalog.append(self._action_entry(action, None, int_by_app, conn_by_app, user_uuid))
        return catalog

    def _action_entry(self, action: Dict[str, Any], detail: Optional[Dict[str, Any]], int_by_app: Dict[str, List[Dict[str, Any]]], conn_by_app: Dict[str, List[Dict[str, Any]]], user_uuid: Optional[str]) -> Dict[str, Any]:
        action_name = normalize_action_name(action)
        app_key = normalize_app_key(action)
        params = (detail or action).get("parameters") or (detail or action).get("params") or (detail or action).get("input_parameters")
        response_schema = (detail or action).get("response") or (detail or action).get("response_schema") or (detail or action).get("output_schema")
        integrations = int_by_app.get(app_key, [])
        connected_accounts = conn_by_app.get(app_key, [])
        requires_auth = bool(integrations)
        execution_ready = (len(connected_accounts) > 0) if (user_uuid is not None) else None

        entry = {
            "action_name": action_name,
            "display_name": action.get("displayName") or action.get("display_name") or action.get("title"),
            "description": action.get("description"),
            "app_key": app_key,
            "app_name": action.get("appName") or action.get("app_name"),
            "enabled": action.get("enabled", True),
            "tags": action.get("tags") or action.get("categories") or [],
            "parameters": params,
            "response_schema": response_schema,
            "integrations": integrations,
            "connected_accounts": connected_accounts,
            "requires_auth": requires_auth,
            "execution_ready": execution_ready,
        }
        if detail:
            entry["full_detail"] = detail
        return entry

    def build_apps_catalog(self, user_uuid: Optional[str] = None, per_page: int = 50, max_pages: int = 3) -> List[Dict[str, Any]]:
        log.info("Fetching apps...")
        apps = self.list_apps(per_page=per_page, max_pages=max_pages)
        log.info("Found %d apps", len(apps))

        log.info("Fetching integrations...")
        integrations = self.list_integrations(per_page=per_page, max_pages=max_pages)
        log.info("Fetching connected accounts...")
        connected = self.list_connected_accounts(user_uuid, per_page=per_page, max_pages=max_pages) if user_uuid else []

        int_by_app: Dict[str, List[Dict[str, Any]]] = {}
        for inte in integrations:
            k = normalize_app_key(inte)
            if k:
                int_by_app.setdefault(k, []).append(inte)

        conn_by_app: Dict[str, List[Dict[str, Any]]] = {}
        for c in connected:
            k = normalize_app_key(c)
            if k:
                conn_by_app.setdefault(k, []).append(c)

        catalog: List[Dict[str, Any]] = []
        for app in apps:
            key = normalize_app_key(app) or app.get("key") or app.get("id")
            catalog.append({
                "app_key": key,
                "app_name": app.get("name"),
                "description": app.get("description"),
                "logo": app.get("logo"),
                "categories": app.get("categories", []),
                "auth_schemes": app.get("auth_schemes") or app.get("auth") or [],
                "integrations": int_by_app.get(key, []),
                "connected_accounts": conn_by_app.get(key, []),
                "is_configured": len(int_by_app.get(key, [])) > 0,
                "is_connected": (len(conn_by_app.get(key, [])) > 0) if user_uuid else None,
                "raw": app,
            })
        return catalog


# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser(prog="composio_inspector")
    p.add_argument("-m", "--mode", choices=("actions", "apps"), default="actions")
    p.add_argument("-u", "--user", dest="user_uuid")
    p.add_argument("-a", "--app", dest="app_filter")
    p.add_argument("-o", "--output", dest="output_file")
    p.add_argument("--per-page", type=int, default=50, help="Items per page when paginating (default 50)")
    p.add_argument("--max-pages", type=int, default=3, help="Max pages to fetch (default 3)")
    p.add_argument("-d", "--detailed", action="store_true", help="Fetch detailed action info (slower)")
    p.add_argument("--workers", type=int, default=4, help="Workers for detailed fetch")
    p.add_argument("--limit", type=int, default=None, help="Ask server for a limit on items (best-effort)")
    p.add_argument("--dry-run", action="store_true", help="Do not write large JSON; print counts and per-app summary")
    p.add_argument("--summary-only", action="store_true", help="Write a compact summary JSON/CSV instead of full payload")
    p.add_argument("--fail-on-missing-auth", action="store_true", help="Exit non-zero if any action requires auth but has no connected account")
    return p.parse_args()


def summarize_catalog(catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(catalog)
    requires_auth = [c for c in catalog if c.get("requires_auth")]
    missing_exec = [c for c in requires_auth if not c.get("execution_ready")]
    per_app: Dict[str, Dict[str, int]] = {}
    for c in catalog:
        ak = c.get("app_key") or "UNKNOWN"
        st = per_app.setdefault(ak, {"count": 0, "requires_auth": 0, "not_exec_ready": 0})
        st["count"] += 1
        if c.get("requires_auth"):
            st["requires_auth"] += 1
        if c.get("requires_auth") and not c.get("execution_ready"):
            st["not_exec_ready"] += 1
    return {
        "total": total,
        "requires_auth_total": len(requires_auth),
        "missing_execution_total": len(missing_exec),
        "per_app": per_app
    }


def write_summary_json(path: str, catalog: List[Dict[str, Any]]):
    s = summarize_catalog(catalog)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    log.info("Wrote summary JSON to %s", path)


def write_summary_csv(path: str, catalog: List[Dict[str, Any]]):
    # CSV: action_name,app_key,requires_auth,execution_ready
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action_name", "app_key", "requires_auth", "execution_ready"])
        for c in catalog:
            w.writerow([c.get("action_name"), c.get("app_key"), bool(c.get("requires_auth")), bool(c.get("execution_ready"))])
    log.info("Wrote summary CSV to %s", path)


def main():
    args = parse_args()
    inspector = ComposioInspector()

    if args.mode == "actions":
        catalog = inspector.build_actions_catalog(
            user_uuid=args.user_uuid,
            app_filter=args.app_filter,
            detailed=args.detailed,
            workers=args.workers,
            per_page=args.per_page,
            max_pages=args.max_pages,
            limit=args.limit
        )
    else:
        catalog = inspector.build_apps_catalog(user_uuid=args.user_uuid, per_page=args.per_page, max_pages=args.max_pages)

    # If dry-run: print summary and exit (no file writes)
    summary = summarize_catalog(catalog)
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        # also print per-app totals for quick scanning
        print("\nPer-app breakdown (sample):")
        for k, v in summary["per_app"].items():
            print(f"  {k}: count={v['count']}, requires_auth={v['requires_auth']}, not_exec_ready={v['not_exec_ready']}")
        # set return code according to fail-on-missing-auth if requested
        if args.fail_on_missing_auth and summary["missing_execution_total"] > 0:
            log.error("Failing due to missing auth execution readiness (count=%d)", summary["missing_execution_total"])
            sys.exit(2)
        sys.exit(0)

    # write output
    if args.summary_only:
        # prefer JSON summary; also write CSV companion (if output provided)
        out_path = args.output_file or "actions_summary.json"
        write_summary_json(out_path, catalog)
        # also produce CSV alongside if user asked for .csv or default
        csv_path = os.path.splitext(out_path)[0] + ".csv"
        write_summary_csv(csv_path, catalog)
        if args.fail_on_missing_auth and summary["missing_execution_total"] > 0:
            log.error("Failing due to missing auth execution readiness (count=%d)", summary["missing_execution_total"])
            sys.exit(2)
        sys.exit(0)

    # default: full payload write (same as before), but warn if large
    payload = json.dumps(catalog, indent=2, ensure_ascii=False)
    if len(payload) > 2000000:  # 2MB threshold heuristic
        log.warning("Result payload is large (%.1f MB). Consider using --summary-only or --dry-run to avoid huge files.", len(payload)/1024/1024)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(payload)
        log.info("Catalog written to %s", args.output_file)
    else:
        MAX = 8000
        if len(payload) > MAX:
            print(payload[:MAX] + "\n... (truncated, use --output to save full)", file=sys.stdout)
            log.info("Output truncated (length=%d). Use --output to save full result.", len(payload))
        else:
            print(payload)

    # final CI fail check
    if args.fail_on_missing_auth and summary["missing_execution_total"] > 0:
        log.error("Failing due to missing auth execution readiness (count=%d)", summary["missing_execution_total"])
        sys.exit(2)


if __name__ == "__main__":
    main()
