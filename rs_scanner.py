"""
RS Scanner (Relative Strength Scanner) v5
==========================================
対象   : 日経225構成銘柄
指標   : RS vs 日経225 / TOPIX（5日・10日・20日）
パターン:
  [A] RS水準   … いずれかの期間で RS > 1.15
  [B] RS転換   … 直近3日でRSが底打ち反転
  [C] 隠れ強気 … 株価5日リターン < 0 かつ RS20日 > 1.10
セクター出遅れ（BNF流）:
  [S] セクター出遅れ … セクター平均5日リターン > 閾値
                       かつ個別銘柄が同セクター内で出遅れ
                       かつ出来高が直近平均より増加
季節性フィルター（バックテスト2015-2025年で検証済み）:
  除外月: 3月・6月・7月（平均リターンがマイナスの月）
  ※ 除外月はシグナルが出ても通知しない
通知   : Discord Webhook
"""

import os
import time
import datetime
import traceback
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")

RS_LEVEL_THRESHOLD  = 1.15
RS_HIDDEN_THRESHOLD = 1.10
RS_PERIODS          = [5, 10, 20]
TOP_N               = 5
FETCH_PERIOD        = "3mo"
SLEEP_SEC           = 1.2

# ──────────────────────────────────────────────
# セクター出遅れ設定（BNF流）
# ──────────────────────────────────────────────
SECTOR_RETURN_THRESHOLD = 0.02
SECTOR_LAG_RATIO        = 0.50
VOLUME_INCREASE_RATIO   = 1.10
SECTOR_TOP_N            = 8

# ──────────────────────────────────────────────
# 季節性フィルター（バックテスト2015-2025検証済み）
# ──────────────────────────────────────────────
MONTHLY_BIAS: dict[int, int] = {
    1:  0,
    2:  0,
    3: -1,
    4:  1,
    5:  1,
    6: -1,
    7: -1,
    8:  0,
    9:  1,
    10: 1,
    11: 1,
    12: 0,
}
EXCLUDE_MONTHS = [m for m, b in MONTHLY_BIAS.items() if b == -1]

# ──────────────────────────────────────────────
# universe.csv 読み込み
# ──────────────────────────────────────────────
UNIVERSE_CSV = Path(__file__).parent / "universe230.csv"

def load_universe(csv_path: Path):
    if not csv_path.exists():
        print(f"[WARN] {csv_path} が見つかりません。")
        return [], {}, {}

    for enc in ("cp932", "utf-8", "utf-8-sig"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"[WARN] {csv_path} の読み込みに失敗しました。")
        return [], {}, {}

    df.columns = [c.strip() for c in df.columns]
    tickers  = df["ticker"].str.strip().tolist()
    name_map = dict(zip(df["ticker"].str.strip(), df["name"].str.strip()))

    MERGE = {
        "電気機器": "電機・精密", "電気機器・精密機器": "電機・精密", "精密機器": "電機・精密",
        "機械": "機械",
        "輸送用機器": "自動車", "輸送用機器・機械": "自動車",
        "化学": "化学", "化学・医薬品": "化学", "ガラス・土石製品": "化学",
        "ゴム製品": "化学", "繊維製品": "化学", "パルプ・紙": "化学",
        "素材・その他製造": "化学", "石油・石炭製品": "化学",
        "医薬品": "医薬品",
        "情報・通信業": "情報・通信", "情報・通信・サービス": "情報・通信",
        "銀行業": "銀行", "銀行・金融・保険": "銀行",
        "保険業": "保険・証券", "証券・商品先物取引業": "保険・証券", "その他金融業": "保険・証券",
        "卸売業": "商社・卸売", "商社・小売・卸売": "商社・卸売",
        "小売業": "小売",
        "鉄鋼": "鉄鋼・素材", "非鉄金属": "鉄鋼・素材", "鉱業": "鉄鋼・素材",
        "建設業": "建設・不動産", "不動産業": "建設・不動産",
        "陸運業": "陸運・物流", "運輸・物流": "陸運・物流", "倉庫・運輸関連業": "陸運・物流",
        "海運業": "海運・空運", "空運業": "海運・空運",
        "食料品": "食品", "食品・農林水産": "食品", "水産・農林業": "食品",
        "電気・ガス業": "電気・ガス",
        "サービス業": "サービス・その他", "その他製品": "サービス・その他",
    }
    df["sector_merged"] = df["sector"].str.strip().map(MERGE).fillna(df["sector"].str.strip())

    sector_map: dict[str, list[str]] = {}
    for sector, grp in df.groupby("sector_merged"):
        sector_map[sector] = sorted(grp["ticker"].str.strip().tolist())

    return tickers, sector_map, name_map

NIKKEI225_SAMPLE, SECTOR_MAP, NAME_MAP = load_universe(UNIVERSE_CSV)

BENCHMARKS = {
    "N225":  "^N225",
    "TOPIX": "1306.T",
}


# ──────────────────────────────────────────────
# セクター出遅れ検出（BNF流）
# ──────────────────────────────────────────────

def calc_sector_ret(tickers: list[str], stock_data: dict[str, pd.Series], period: int = 5) -> float | None:
    rets = []
    for t in tickers:
        s = stock_data.get(t)
        if s is not None and len(s) > period:
            ret = (s.iloc[-1] - s.iloc[-(period + 1)]) / s.iloc[-(period + 1)]
            rets.append(ret)
    if not rets:
        return None
    return float(np.mean(rets))


def detect_sector_laggards(
    stock_data: dict[str, pd.Series],
    volume_data: dict[str, pd.DataFrame],
    scan_date: str,
) -> list[dict]:
    results = []

    for sector_name, tickers in SECTOR_MAP.items():
        available = [t for t in tickers if t in stock_data]
        if len(available) < 2:
            continue

        sector_ret = calc_sector_ret(available, stock_data, period=5)
        if sector_ret is None or sector_ret < SECTOR_RETURN_THRESHOLD:
            continue

        for ticker in available:
            s = stock_data[ticker]
            if len(s) < 6:
                continue

            try:
                v_last = float(s.iloc[-1])
                v_prev = float(s.iloc[-6])
                if v_prev == 0 or np.isnan(v_last) or np.isnan(v_prev):
                    continue
                stock_ret = (v_last - v_prev) / v_prev
            except Exception:
                continue

            if np.isnan(stock_ret):
                continue

            if stock_ret >= sector_ret * SECTOR_LAG_RATIO:
                continue

            vol_ratio = None
            vdf = volume_data.get(ticker)
            if vdf is not None and "Volume" in vdf.columns and len(vdf) >= 21:
                try:
                    vol_series  = vdf["Volume"].squeeze()
                    recent_vol  = float(vol_series.iloc[-1])
                    avg_vol     = float(vol_series.iloc[-21:-1].mean())
                    if avg_vol > 0 and not np.isnan(recent_vol) and not np.isnan(avg_vol):
                        vol_ratio = recent_vol / avg_vol
                        if vol_ratio < VOLUME_INCREASE_RATIO:
                            continue
                except Exception:
                    pass

            lag_score = round((sector_ret - stock_ret) * 100, 2)

            try:
                close_val = round(float(s.iloc[-1]), 0)
            except Exception:
                continue

            results.append({
                "ticker":       ticker,
                "sector":       sector_name,
                "sector_ret5d": round(sector_ret * 100, 2),
                "stock_ret5d":  round(stock_ret * 100, 2),
                "lag_score":    lag_score,
                "volume_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
                "close":        close_val,
            })

    results.sort(key=lambda x: x["lag_score"], reverse=True)
    return results[:SECTOR_TOP_N]


# ──────────────────────────────────────────────
# データ取得
# ──────────────────────────────────────────────
def fetch_close(ticker: str) -> pd.Series | None:
    try:
        df = yf.download(ticker, period=FETCH_PERIOD, progress=False, auto_adjust=True)
        if df.empty or len(df) < 25:
            return None
        return df["Close"].squeeze()
    except Exception:
        return None


def fetch_ohlcv_all(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """High/Low/Close/Volume を一括取得（ATR計算に使用）"""
    BATCH = 50
    data: dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        try:
            raw = yf.download(
                batch,
                period=FETCH_PERIOD,
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
            for t in batch:
                try:
                    if len(batch) == 1:
                        df = raw[["High", "Low", "Close", "Volume"]].copy()
                    else:
                        df = raw[t][["High", "Low", "Close", "Volume"]].copy()
                    if df is not None and len(df) >= 25:
                        data[t] = df
                except Exception:
                    pass
        except Exception as e:
            print(f"  [WARN] バッチ取得失敗 ({batch[0]}〜): {e}")
        time.sleep(SLEEP_SEC)

    return data


# ──────────────────────────────────────────────
# RS計算
# ──────────────────────────────────────────────
def calc_rs(stock: pd.Series, bench: pd.Series, period: int) -> pd.Series:
    common = stock.index.intersection(bench.index)
    s = stock.loc[common]
    b = bench.loc[common]
    s_ret = s.pct_change(period)
    b_ret = b.pct_change(period)
    return s_ret / b_ret.replace(0, np.nan)


# ──────────────────────────────────────────────
# 売買水準計算（ATRベース参考値）
# ──────────────────────────────────────────────
def calc_rs_levels(close: float, atr: float) -> dict:
    return {
        "entry_low":  round(close - atr * 0.3),
        "entry_high": round(close + atr * 0.3),
        "stop_loss":  round(close - atr * 1.5),
        "target":     round(close + atr * 3.0),
    }


# ──────────────────────────────────────────────
# パターン検出
# ──────────────────────────────────────────────
def detect_patterns(
    ticker: str,
    stock: pd.Series,
    bench_data: dict[str, pd.Series],
) -> dict | None:
    signals    = []
    score      = 0
    rs_snapshot = {}
    price_ret_5 = stock.pct_change(5).iloc[-1]

    for bname, bench in bench_data.items():
        for period in RS_PERIODS:
            rs = calc_rs(stock, bench, period)
            if rs.dropna().empty:
                continue

            rs_latest = rs.iloc[-1]
            rs_snapshot[f"RS{period}_{bname}"] = round(float(rs_latest), 3)

            if rs_latest > RS_LEVEL_THRESHOLD:
                signals.append(f"[A] RS水準 RS{period}({bname})={rs_latest:.2f}")
                score += 1

            if len(rs.dropna()) >= 4:
                rs_tail = rs.dropna().iloc[-4:]
                min_idx = rs_tail.iloc[:-1].argmin()
                if min_idx >= 1 and rs_tail.iloc[-1] > rs_tail.iloc[-2]:
                    if rs_tail.iloc[min_idx] < rs_tail.iloc[0]:
                        signals.append(f"[B] RS転換 RS{period}({bname}) 底打ち反転")
                        score += 1

            if period == 20:
                if price_ret_5 < 0 and rs_latest > RS_HIDDEN_THRESHOLD:
                    signals.append(
                        f"[C] 隠れ強気 RS20({bname})={rs_latest:.2f} 株価5日{price_ret_5*100:.1f}%"
                    )
                    score += 2

    if score == 0:
        return None

    return {
        "ticker":      ticker,
        "score":       score,
        "signals":     list(dict.fromkeys(signals)),
        "rs_values":   rs_snapshot,
        "price_ret_5": price_ret_5,
        "close":       float(stock.iloc[-1]),
    }


# ──────────────────────────────────────────────
# Discord通知
# ──────────────────────────────────────────────
SIG_BADGE = {"[A]": "A", "[B]": "B", "[C]": "C"}

def format_discord_embeds(results: list[dict], scan_date: str) -> list[dict]:
    if not results:
        return [{
            "title": f"📈 RSスキャナー — {scan_date}",
            "description": "本日シグナルなし",
            "color": 0x555555,
        }]

    lines = []
    for r in results:
        ret5 = r["price_ret_5"] * 100
        badges = "".join(
            f"[{SIG_BADGE[k]}]"
            for k in SIG_BADGE
            if any(k in s for s in r["signals"])
        )
        rs20 = r["rs_values"].get("RS20_N225", "-")
        icon = "🔥" if r["score"] >= 6 else "⚡"
        name = NAME_MAP.get(r["ticker"], r["ticker"])
        line = (
            f"{icon} **{name}**（{r['ticker']}） {badges}  "
            f"{r['close']:,.0f}円  {ret5:+.1f}%  RS20={rs20}"
        )
        if r.get("entry_low"):
            line += (
                f"\n　 📌 参考: {r['entry_low']:,}〜{r['entry_high']:,}円"
                f" | 🛑 撤退目安: {r['stop_loss']:,}円"
                f" | 🎯 目標目安: {r['target']:,}円"
            )
        lines.append(line)

    bias_label = "強気月" if MONTHLY_BIAS.get(datetime.date.today().month, 0) > 0 else "中立月"
    description = "\n".join(lines)
    description += (
        f"\n\n対象{len(NIKKEI225_SAMPLE)}銘柄 / "
        f"検出{len(results)}件→上位{len(results[:TOP_N])}件表示 / "
        f"{bias_label}"
    )

    return [{
        "title": f"📈 RSスキャナー — {scan_date}",
        "description": description,
        "color": 0x1a6b9a,
    }]


def send_discord(results: list[dict], scan_date: str):
    if not DISCORD_WEBHOOK:
        print("[WARN] DISCORD_WEBHOOK 未設定")
        return
    embeds = format_discord_embeds(results, scan_date)
    payload = {"embeds": embeds}
    resp = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
    if resp.status_code not in (200, 204):
        print(f"[ERROR] Discord送信失敗: {resp.status_code}")
    print(f"[OK] Discord通知完了（{len(results)}件）")


def send_discord_no_signal(scan_date: str):
    if not DISCORD_WEBHOOK:
        return
    payload = {"embeds": [{
        "title": f"📈 RSスキャナー — {scan_date}",
        "description": "本日シグナルなし",
        "color": 0x555555,
    }]}
    requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)


def send_discord_sector(sector_results: list[dict], scan_date: str):
    if not DISCORD_WEBHOOK or not sector_results:
        return

    lines = []
    for r in sector_results[:SECTOR_TOP_N]:
        vol_str = f"{r['volume_ratio']}倍" if r["volume_ratio"] else "-"
        name = NAME_MAP.get(r["ticker"], r["ticker"])
        lag  = r["lag_score"]
        lag_str = f"{lag:+.1f}%pt" if lag == lag else "-"
        lines.append(
            f"⏳ **{name}**（{r['ticker']}） [{r['sector']}]  "
            f"{r['close']:,.0f}円  "
            f"出遅れ{lag_str}  "
            f"出来高{vol_str}"
        )

    payload = {"embeds": [{
        "title": f"🏭 セクター出遅れ — {scan_date}",
        "description": "\n".join(lines),
        "color": 0x8e44ad,
    }]}
    resp = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
    if resp.status_code not in (200, 204):
        print(f"[ERROR] Discord送信失敗（セクター）: {resp.status_code}")
    print(f"[OK] セクター出遅れ Discord通知完了（{len(sector_results)}件）")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main():
    scan_date = datetime.date.today().isoformat()
    today     = datetime.date.today()
    print(f"\n{'='*50}")
    print(f"RS Scanner v5 開始: {scan_date}")
    print(f"{'='*50}")

    current_month = today.month
    monthly_bias  = MONTHLY_BIAS.get(current_month, 0)
    if monthly_bias < 0:
        msg = (
            f"⚠️ 季節性フィルター: {current_month}月は除外月です "
            f"（バックテスト平均リターン: マイナス）。\n"
            f"シグナルスキャンをスキップします。"
        )
        print(f"\n{msg}")
        if DISCORD_WEBHOOK:
            payload = {"embeds": [{
                "title": f"📅 RSスキャナー — {scan_date}",
                "description": msg,
                "color": 0x888780,
            }]}
            requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        return

    bias_label = "強気月" if monthly_bias > 0 else "中立月"
    print(f"\n季節性チェック: {current_month}月 = {bias_label} → スキャン続行")

    print("\n[1/4] ベンチマーク取得中...")
    bench_data = {}
    for bname, bticker in BENCHMARKS.items():
        s = fetch_close(bticker)
        if s is not None:
            bench_data[bname] = s
            print(f"  {bname}: {len(s)}日分 OK")
        time.sleep(SLEEP_SEC)

    if not bench_data:
        print("[ERROR] ベンチマーク取得失敗")
        return

    print(f"\n[2/4] 銘柄データ一括取得中（{len(NIKKEI225_SAMPLE)}銘柄）...")
    ohlcv_data = fetch_ohlcv_all(NIKKEI225_SAMPLE)
    print(f"  取得成功: {len(ohlcv_data)}/{len(NIKKEI225_SAMPLE)}銘柄")

    stock_data:  dict[str, pd.Series]    = {}
    volume_data: dict[str, pd.DataFrame] = {}
    for t, df in ohlcv_data.items():
        stock_data[t]  = df["Close"].squeeze()
        volume_data[t] = df

    print("\n[3/4] RSスキャン実行中...")
    results = []
    for ticker, stock in stock_data.items():
        r = detect_patterns(ticker, stock, bench_data)
        if r:
            results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:TOP_N]
    print(f"  RS検出: {len(results)}件 → 上位{len(top_results)}件")

    # ATR・売買水準を追加
    for r in top_results:
        ohlcv = volume_data.get(r["ticker"])
        if ohlcv is not None and "High" in ohlcv.columns and len(ohlcv) >= 14:
            atr = float((ohlcv["High"] - ohlcv["Low"]).rolling(14).mean().iloc[-1])
            r.update(calc_rs_levels(r["close"], atr))

    print("\n[4/4] セクター出遅れスキャン実行中...")
    sector_results = detect_sector_laggards(stock_data, volume_data, scan_date)
    print(f"  セクター出遅れ検出: {len(sector_results)}件")

    if top_results:
        send_discord(top_results, scan_date)
    else:
        send_discord_no_signal(scan_date)

    if sector_results:
        send_discord_sector(sector_results, scan_date)

    print(f"\n── RSシグナル上位 ──")
    for r in top_results:
        print(f"  {r['ticker']:10s} score:{r['score']}  {r['signals'][0] if r['signals'] else ''}")

    print(f"\n── セクター出遅れ上位 ──")
    for r in sector_results:
        print(f"  {r['ticker']:10s} [{r['sector']}]  出遅れ{r['lag_score']:+.1f}%pt")

    print(f"\n{'='*50}")
    print("完了")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
