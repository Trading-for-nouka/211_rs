"""
monitor.py — 211_rs ポジション監視・エグジット通知
=================================================
実行タイミング: 毎営業日 15:30以降（取引終了後）
               GitHub Actions（.github/workflows/monitor.yml）で自動実行

機能:
  1. GitHub の 211_rs/positions.json からポジション読み込み
  2. yfinance で現在価格を取得
  3. エグジット条件チェック:
       - 保有20営業日到達 → 利確検討通知
       - 現在値 ≤ 損切りライン → 損切り通知
  4. Discord にエグジット通知（個別 + サマリー）
  5. エグジット対象を positions.json から削除し GitHub に書き戻し
  6. 継続保有ポジションの最高値（highest_price）を更新

301_portfolio_summary との連動フロー:
  rs_scanner2.py → Discord通知（📎形式）
      ↓ ユーザーが 🛒 リアクション
  discord_poll.py → positions.json に追加（trading-for-nouka/211_rs）
      ↓ 毎営業日
  monitor.py → エグジット判定 → 通知 → positions.json から削除

必要な環境変数（.env）:
  DISCORD_WEBHOOK  : Discord通知用Webhook URL
  PAT_TOKEN        : GitHub Personal Access Token（positions.json読み書き用）
"""

import json
import os
import base64
import time
from datetime import datetime, timezone, timedelta

import requests
import yfinance as yf
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── 設定 ──────────────────────────────────────────────
DISCORD_WEBHOOK  = os.environ.get("DISCORD_WEBHOOK", "")
PAT_TOKEN        = os.environ.get("PAT_TOKEN", "")
GITHUB_REPO      = "trading-for-nouka/211_rs"
POSITIONS_FILE   = "rs_positions.json"

HOLD_DAYS_TARGET = 20     # バックテスト最適保有営業日数
SLEEP_SEC        = 0.5

GH_HEADERS = {
    "Authorization": f"token {PAT_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}
# ──────────────────────────────────────────────────────


# ── GitHub API ────────────────────────────────────────

def get_github_positions() -> tuple[list, str | None]:
    """positions.json を GitHub から取得。存在しない場合は空リストを返す。"""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{POSITIONS_FILE}"
    r = requests.get(url, headers=GH_HEADERS, timeout=10)
    if r.status_code == 404:
        return [], None
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API error: {r.status_code} — {r.text[:200]}")
    d = r.json()
    positions = json.loads(base64.b64decode(d["content"]).decode())
    return positions, d["sha"]


def put_github_positions(positions: list, sha: str | None, msg: str) -> bool:
    """positions.json を GitHub に書き戻す。"""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{POSITIONS_FILE}"
    encoded = base64.b64encode(
        json.dumps(positions, ensure_ascii=False, indent=2).encode()
    ).decode()
    payload = {"message": msg, "content": encoded}
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=GH_HEADERS, json=payload, timeout=10)
    return r.status_code in (200, 201)


# ── 価格取得・日数計算 ─────────────────────────────────

def get_jst_today():
    return datetime.now(timezone(timedelta(hours=9))).date()


def count_trading_days(entry_date_str: str) -> int:
    """エントリー日から今日までの営業日数（日本の祝日は未考慮の近似値）"""
    entry = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
    today = get_jst_today()
    return max(0, int(np.busday_count(entry, today)))


def get_current_prices(tickers: list) -> dict:
    """yfinance で現在の終値を取得する。"""
    prices = {}
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period="2d", auto_adjust=True)
            if not df.empty:
                prices[t] = float(df["Close"].iloc[-1])
        except Exception as e:
            print(f"  [WARN] {t} 価格取得失敗: {e}")
        time.sleep(SLEEP_SEC)
    return prices


# ── Discord 通知 ──────────────────────────────────────

def send_exit_notification(pos: dict, current_price: float, reason: str, pnl_pct: float):
    """個別エグジット通知を Discord に送信する。"""
    if not DISCORD_WEBHOOK:
        return
    color = 0x2ecc71 if pnl_pct >= 0 else 0xe74c3c
    icon  = "🟢" if pnl_pct >= 0 else "🔴"
    days  = count_trading_days(pos["entry_date"])
    payload = {"embeds": [{
        "title": f"[211_rs] 📤 エグジット通知 — {pos['name']}（{pos['ticker']}）",
        "description": (
            f"{icon} **{reason}**\n\n"
            f"　 エントリー : {pos['entry_price']:,.0f}円"
            f"（{pos['entry_date']}）\n"
            f"　 現在値     : {current_price:,.0f}円\n"
            f"　 損益       : **{pnl_pct:+.1f}%**\n"
            f"　 保有日数   : {days}営業日\n"
            f"　 最高値     : {pos.get('highest_price', pos['entry_price']):,.0f}円"
        ),
        "color": color,
    }]}
    resp = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
    if resp.status_code not in (200, 204):
        print(f"  [ERROR] Discord送信失敗: {resp.status_code}")


def send_hold_summary(summary_lines: list, today_str: str):
    """継続保有ポジションの一覧を Discord に送信する。"""
    if not DISCORD_WEBHOOK or not summary_lines:
        return
    payload = {"embeds": [{
        "title": f"[211_rs] 📊 保有状況 — {today_str}",
        "description": "\n".join(summary_lines),
        "color": 0x1a6b9a,
    }]}
    requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)


# ── メイン ────────────────────────────────────────────

def main():
    today_str = get_jst_today().isoformat()
    print(f"\n{'='*52}")
    print(f"  211_rs ポジション監視 — {today_str}")
    print(f"{'='*52}")

    if not PAT_TOKEN:
        print("[ERROR] PAT_TOKEN が未設定です（.env を確認してください）")
        return

    # ── Step 1: ポジション読み込み ──────────────────────
    print("\n[1/3] positions.json 読み込み中...")
    try:
        positions, sha = get_github_positions()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    if not positions:
        print("  保有ポジションなし")
        return
    print(f"  {len(positions)}件のポジションを確認")

    # ── Step 2: 現在価格取得 ────────────────────────────
    print(f"\n[2/3] 現在価格取得中（{len(positions)}銘柄）...")
    tickers = [p["ticker"] for p in positions]
    prices  = get_current_prices(tickers)
    print(f"  取得成功: {len(prices)}/{len(tickers)}銘柄")

    # ── Step 3: エグジット判定 ──────────────────────────
    print("\n[3/3] エグジット判定中...")
    remaining     = []
    exit_count    = 0
    summary_lines = []
    updated       = False

    for pos in positions:
        ticker  = pos["ticker"]
        current = prices.get(ticker)

        if current is None:
            print(f"  [SKIP] {ticker}（{pos.get('name','')}）価格取得不可")
            remaining.append(pos)
            continue

        entry = pos["entry_price"]
        stop  = pos.get("stop_loss", 0)
        pnl   = (current - entry) / entry * 100
        days  = count_trading_days(pos["entry_date"])

        # 最高値更新
        prev_high = pos.get("highest_price", entry)
        if current > prev_high:
            pos["highest_price"] = round(current, 0)
            updated = True

        # エグジット判定
        exit_reason = None
        if stop > 0 and current <= stop:
            exit_reason = f"損切りライン到達（{stop:,.0f}円）"
        elif days >= HOLD_DAYS_TARGET:
            exit_reason = f"保有{days}営業日（目標{HOLD_DAYS_TARGET}日到達）"

        if exit_reason:
            send_exit_notification(pos, current, exit_reason, pnl)
            print(f"  [EXIT] {ticker:10s} {pos.get('name',''):12s}"
                  f" → {exit_reason}  損益{pnl:+.1f}%")
            exit_count += 1
            updated = True
        else:
            remaining.append(pos)
            icon = "🟢" if pnl >= 0 else "🔴"
            summary_lines.append(
                f"{icon} **{pos.get('name', ticker)}**（{ticker}）"
                f"  {current:,.0f}円  {pnl:+.1f}%  {days}日目"
            )
            print(f"  [HOLD] {ticker:10s} {pos.get('name',''):12s}"
                  f"  {days:>2}日  {pnl:+.1f}%  現在{current:,.0f}円")

    # ── GitHub への書き戻し ─────────────────────────────
    if updated:
        ok = put_github_positions(
            remaining, sha,
            f"monitor: {exit_count}件エグジット処理 / {today_str}"
        )
        print(f"\n  positions.json 更新: {'OK ✅' if ok else 'FAILED ❌'}")
    else:
        print("\n  positions.json: 変更なし")

    # ── 保有状況サマリーを Discord に送信 ──────────────
    if summary_lines:
        send_hold_summary(summary_lines, today_str)

    print(f"\n{'='*52}")
    print(f"  完了: EXIT {exit_count}件 / 継続保有 {len(remaining)}件")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[FATAL] {e}")
        traceback.print_exc()
