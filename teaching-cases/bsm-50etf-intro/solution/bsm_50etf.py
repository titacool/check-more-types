"""上证 50ETF 认购期权的 Black-Scholes-Merton 入门参考答案。

解决的问题
    在剩余不少于 7 个自然日的最近到期月份里，选出最接近平值的认购合约，
    用历史波动率、Shibor 和分红信息计算理论价，并判断市场最新价相对模型
    是偏贵还是偏便宜。

运行
    python bsm_50etf.py
    python bsm_50etf.py --check-only
"""

from __future__ import annotations

import argparse
import math
import re
from datetime import date, datetime, timedelta

import akshare as ak
import pandas as pd


UNDERLYING = "510050"
UNDERLYING_SINA = "sh510050"
OPTION_NAME = "华夏上证50ETF期权"
LIST_SYMBOL = "50ETF"
CONTRACT_MULTIPLIER = 10_000
HISTORY_WINDOW = 252
CALENDAR_DAYS = 365
MIN_CALENDAR_DAYS = 7
# Shibor 报价是年化百分比。代表天数用来选择“和剩余期限最接近”的一档。
SHIBOR_TENORS = (
    ("O/N", "O/N-定价", 1),
    ("1W", "1W-定价", 7),
    ("2W", "2W-定价", 14),
    ("1M", "1M-定价", 30),
    ("3M", "3M-定价", 90),
    ("6M", "6M-定价", 180),
    ("9M", "9M-定价", 270),
    ("1Y", "1Y-定价", 365),
)
_CONTRACT_CODE = re.compile(r"^[0-9]+([CP])[0-9]+M[0-9]+$")


def norm_cdf(x: float) -> float:
    """标准正态分布函数，用误差函数写出，避免额外安装 SciPy。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(
    spot: float,
    strike: float,
    year_fraction: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> tuple[float, float]:
    if spot <= 0 or strike <= 0:
        raise ValueError("标的价格和行权价必须为正")
    if year_fraction <= 0:
        raise ValueError("剩余期限必须为正")
    if volatility <= 0:
        raise ValueError("波动率必须为正")
    vol_sqrt_t = volatility * math.sqrt(year_fraction)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * year_fraction
    ) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bsm_price(
    spot: float,
    strike: float,
    year_fraction: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    """Black-Scholes-Merton 欧式认购或认沽价格。option_type 取 call 或 put。"""
    d1, d2 = _d1_d2(spot, strike, year_fraction, rate, volatility, dividend_yield)
    deferred_spot = spot * math.exp(-dividend_yield * year_fraction)
    deferred_strike = strike * math.exp(-rate * year_fraction)
    if option_type == "call":
        return deferred_spot * norm_cdf(d1) - deferred_strike * norm_cdf(d2)
    if option_type == "put":
        return deferred_strike * norm_cdf(-d2) - deferred_spot * norm_cdf(-d1)
    raise ValueError("option_type 只能是 call 或 put")


def bsm_delta(
    spot: float,
    strike: float,
    year_fraction: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    """欧式期权的 Delta。认购约为 e^{-qT} N(d1)，认沽约为 e^{-qT} (N(d1) - 1)。"""
    d1, _d2 = _d1_d2(spot, strike, year_fraction, rate, volatility, dividend_yield)
    discount = math.exp(-dividend_yield * year_fraction)
    if option_type == "call":
        return discount * norm_cdf(d1)
    if option_type == "put":
        return discount * (norm_cdf(d1) - 1.0)
    raise ValueError("option_type 只能是 call 或 put")


def annualized_volatility(closes: pd.Series, window: int = HISTORY_WINDOW) -> float:
    """用最近 window 个对数收益率的样本标准差，按 252 个交易日年化。"""
    prices = closes.astype(float)
    log_returns = (prices / prices.shift(1)).map(math.log).dropna()
    sample = log_returns.iloc[-window:]
    if len(sample) < window:
        raise ValueError(f"有效收益率只有 {len(sample)} 个，不足 {window} 个")
    return float(sample.std(ddof=1) * math.sqrt(HISTORY_WINDOW))


def choose_shibor_tenor(calendar_days: int) -> tuple[str, str]:
    """返回 (期限简称, 列名)。并列时取更短的期限。"""
    if calendar_days < 0:
        raise ValueError("剩余天数不能为负")
    name, column, _days = min(
        SHIBOR_TENORS,
        key=lambda item: (abs(item[2] - calendar_days), item[2]),
    )
    return name, column


def shibor_to_continuous(percent_quote: float) -> float:
    """Shibor 年化百分比报价转成连续复利。1.35 表示 1.35%。"""
    simple = percent_quote / 100.0
    if simple <= -1:
        raise ValueError("Shibor 报价无法转换成连续复利")
    return math.log(1.0 + simple)


def project_next_ex_date(last_ex_date: date, as_of: date) -> date:
    """假设分红月日保持不变，推出严格晚于估值日的下一次除息日。"""
    year = as_of.year
    while True:
        candidate = date(year, last_ex_date.month, last_ex_date.day)
        if candidate > as_of:
            return candidate
        year += 1


def dividend_yield_for_contract(
    dividends: pd.DataFrame,
    spot: float,
    as_of: date,
    expiry: date,
    year_fraction: float,
) -> dict:
    """到期前如果遇不到分红，q = 0；遇得到，就把最近一次现金分红折进剩余期限。"""
    frame = dividends.copy()
    frame["日期"] = pd.to_datetime(frame["日期"]).dt.date
    frame["累计分红"] = pd.to_numeric(frame["累计分红"])
    frame = frame.sort_values("日期")
    if frame.empty:
        raise ValueError("分红表是空的")
    if len(frame) == 1:
        cash = float(frame["累计分红"].iloc[0])
    else:
        cash = float(frame["累计分红"].iloc[-1] - frame["累计分红"].iloc[-2])
    if cash < 0:
        raise ValueError("最近一次累计分红下降，无法估计现金分红")
    last_ex = frame["日期"].iloc[-1]
    next_ex = project_next_ex_date(last_ex, as_of)
    if next_ex > expiry or cash == 0:
        dividend_yield = 0.0
    else:
        if cash >= spot:
            raise ValueError("现金分红不应大于等于标的价格")
        # ln(S / (S - D)) / T，使剩余期限内大约扣除这一笔现金分红。
        dividend_yield = math.log(spot / (spot - cash)) / year_fraction
    return {
        "cash_dividend": cash,
        "last_ex_date": last_ex,
        "next_ex_date": next_ex,
        "dividend_before_expiry": next_ex <= expiry and cash > 0,
        "dividend_yield": dividend_yield,
    }


def option_right(contract_code: str) -> str:
    match = _CONTRACT_CODE.match(str(contract_code).strip())
    if match is None:
        raise ValueError(f"无法识别的合约代码: {contract_code}")
    return match.group(1)


def quoted_premium(row: pd.Series) -> float:
    """最新价优先；最新价为 0 时用前结算价。"""
    last = float(row["当前价"])
    if last > 0:
        return last
    previous = float(row["前结价"])
    if previous > 0:
        return previous
    raise ValueError(f"{row['合约交易代码']} 没有可用的权利金报价")


def pick_nearest_contract(board: pd.DataFrame, spot: float, right: str) -> pd.Series:
    """选出指定权利方向里，行权价最接近现价的一张。距离相同取较低行权价。"""
    tagged = board.copy()
    tagged["权利"] = tagged["合约交易代码"].map(option_right)
    same_right = tagged.loc[tagged["权利"] == right].copy()
    if same_right.empty:
        raise ValueError(f"行情里没有权利方向为 {right} 的合约")
    same_right["距离"] = (same_right["行权价"].astype(float) - spot).abs()
    same_right = same_right.sort_values(["距离", "行权价"])
    return same_right.iloc[0]


def choose_expiry(
    contracts: list[dict], quote_date: date, min_days: int = MIN_CALENDAR_DAYS
) -> dict:
    """contracts 的每个元素含 month（YYYYMM）和 expiry（date）。"""
    usable = []
    for item in contracts:
        days = (item["expiry"] - quote_date).days
        if days >= min_days:
            usable.append({**item, "calendar_days": days})
    if not usable:
        raise ValueError(f"没有剩余不少于 {min_days} 个自然日的到期月份")
    usable.sort(key=lambda item: item["expiry"])
    return usable[0]


def year_fraction(calendar_days: int) -> float:
    return calendar_days / CALENDAR_DAYS


def put_call_parity_gap(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    year_fraction: float,
    rate: float,
    dividend_yield: float,
) -> float:
    """C - P - (S e^{-qT} - K e^{-rT})。模型价格应非常接近 0。"""
    forward_difference = spot * math.exp(-dividend_yield * year_fraction) - strike * math.exp(
        -rate * year_fraction
    )
    return call_price - put_price - forward_difference


def build_conclusion(
    call_model: float,
    call_market: float,
    volatility: float,
    dividend_yield: float,
    tenor_name: str,
) -> str:
    gap = call_model - call_market
    if abs(gap) < 0.001:
        if gap > 0:
            relation = "模型价和市场最新价非常接近，模型略高"
        elif gap < 0:
            relation = "模型价和市场最新价非常接近，模型略低"
        else:
            relation = "模型价和市场最新价几乎相同"
    elif gap > 0:
        relation = "模型价高于市场最新价。按这份历史波动率来看，市场报价更便宜"
    else:
        relation = "模型价低于市场最新价。按这份历史波动率来看，市场报价更贵"
    if dividend_yield == 0:
        dividend_sentence = "这张合约在到期前预计不会遇到分红，所以连续股息率取 0。"
    else:
        dividend_sentence = "到期前预计会遇到一次分红，连续股息率已经把这笔现金分红折进剩余期限。"
    return (
        f"{relation}。"
        "比较时要记住两个近似：模型用的是过去一年已经实现的历史波动率"
        f"（{volatility:.2%}），市场价反映的是对未来波动的预期；"
        f"无风险利率用了剩余期限最接近的 {tenor_name} Shibor，并换成了连续复利。"
        f"{dividend_sentence}"
        "这是一次入门体检，不是交易建议。"
    )


def _require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{source} 缺少字段 {missing}，实际字段为 {list(frame.columns)}")


def _call_data(func, *args, **kwargs):
    last_error = None
    for _attempt in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 行情接口会偶发网络错误，重试后再抛出。
            last_error = exc
    raise RuntimeError(f"调用 {getattr(func, '__name__', func)} 失败") from last_error


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _table_lookup(frame: pd.DataFrame) -> dict:
    _require_columns(frame, ["字段", "值"], "行情对照表")
    return dict(zip(frame["字段"], frame["值"]))


def fetch_underlying_quote() -> dict:
    raw = _call_data(ak.option_sse_underlying_spot_price_sina, symbol=UNDERLYING_SINA)
    quote = _table_lookup(raw)
    spot = float(quote["最近成交价"])
    if spot <= 0:
        spot = float(quote["昨日收盘价"])
    if spot <= 0:
        raise RuntimeError("标的最新价和昨收都不可用")
    return {
        "name": str(quote["证券简称"]),
        "spot": spot,
        "quote_date": _as_date(quote["行情日期"]),
        "quote_time": str(quote["行情时间"]),
    }


def fetch_expiries() -> list[dict]:
    months = _call_data(ak.option_sse_list_sina, symbol=LIST_SYMBOL, exchange="null")
    if not months:
        raise RuntimeError("没有取到 50ETF 期权到期月份")
    contracts = []
    for month in months:
        expire_day, _remainder_days = _call_data(
            ak.option_sse_expire_day_sina,
            trade_date=str(month),
            symbol=LIST_SYMBOL,
            exchange="null",
        )
        # 不用接口返回的剩余天数。那个天数相对“现在”，可能和行情所属交易日不一致。
        contracts.append({"month": str(month), "expiry": _as_date(expire_day)})
    return contracts


def fetch_option_board(yyyymm: str) -> pd.DataFrame:
    end_month = yyyymm[2:] if len(yyyymm) == 6 else yyyymm
    board = _call_data(ak.option_finance_board, symbol=OPTION_NAME, end_month=end_month)
    _require_columns(board, ["日期", "合约交易代码", "当前价", "前结价", "行权价"], "期权行情")
    if board.empty:
        raise RuntimeError(f"{yyyymm} 没有期权行情")
    return board


def fetch_history(quote_date: date) -> pd.Series:
    start = quote_date - timedelta(days=450)
    history = _call_data(
        ak.fund_etf_hist_em,
        symbol=UNDERLYING,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=quote_date.strftime("%Y%m%d"),
        adjust="qfq",
    )
    _require_columns(history, ["日期", "收盘"], "ETF 日线")
    history = history.copy()
    history["日期"] = pd.to_datetime(history["日期"])
    history = history.loc[history["日期"] <= pd.Timestamp(quote_date)]
    history = history.sort_values("日期")
    return history["收盘"]


def fetch_dividends() -> pd.DataFrame:
    dividends = _call_data(ak.fund_etf_dividend_sina, symbol=UNDERLYING_SINA)
    _require_columns(dividends, ["日期", "累计分红"], "ETF 分红")
    return dividends


def fetch_shibor_quote(column: str) -> dict:
    shibor = _call_data(ak.macro_china_shibor_all)
    _require_columns(shibor, ["日期", column], "Shibor")
    frame = shibor.copy()
    frame["日期"] = pd.to_datetime(frame["日期"])
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = frame.dropna(subset=[column]).sort_values("日期")
    if valid.empty:
        raise RuntimeError(f"Shibor 列 {column} 没有有效报价")
    latest = valid.iloc[-1]
    return {"date": latest["日期"].date(), "percent_quote": float(latest[column])}


def price_live_contract() -> dict:
    underlying = fetch_underlying_quote()
    expiries = fetch_expiries()
    selected = choose_expiry(expiries, underlying["quote_date"])
    board = fetch_option_board(selected["month"])
    call = pick_nearest_contract(board, underlying["spot"], "C")
    put = pick_nearest_contract(board, float(call["行权价"]), "P")
    closes = fetch_history(underlying["quote_date"])
    volatility = annualized_volatility(closes)
    dividends = dividend_yield_for_contract(
        fetch_dividends(),
        underlying["spot"],
        underlying["quote_date"],
        selected["expiry"],
        year_fraction(selected["calendar_days"]),
    )
    tenor_name, tenor_column = choose_shibor_tenor(selected["calendar_days"])
    shibor = fetch_shibor_quote(tenor_column)
    rate = shibor_to_continuous(shibor["percent_quote"])
    fraction = year_fraction(selected["calendar_days"])
    strike = float(call["行权价"])
    call_market = quoted_premium(call)
    put_market = quoted_premium(put)
    call_model = bsm_price(
        underlying["spot"], strike, fraction, rate, volatility, dividends["dividend_yield"], "call"
    )
    put_model = bsm_price(
        underlying["spot"], strike, fraction, rate, volatility, dividends["dividend_yield"], "put"
    )
    call_delta = bsm_delta(
        underlying["spot"], strike, fraction, rate, volatility, dividends["dividend_yield"], "call"
    )
    return {
        "underlying": underlying,
        "candidates": expiries,
        "selected": selected,
        "call": call,
        "put": put,
        "strike": strike,
        "volatility": volatility,
        "dividends": dividends,
        "tenor_name": tenor_name,
        "shibor": shibor,
        "rate": rate,
        "year_fraction": fraction,
        "call_market": call_market,
        "put_market": put_market,
        "call_model": call_model,
        "put_model": put_model,
        "call_delta": call_delta,
        "model_parity_gap": put_call_parity_gap(
            call_model,
            put_model,
            underlying["spot"],
            strike,
            fraction,
            rate,
            dividends["dividend_yield"],
        ),
        "market_parity_gap": put_call_parity_gap(
            call_market,
            put_market,
            underlying["spot"],
            strike,
            fraction,
            rate,
            dividends["dividend_yield"],
        ),
    }


def _yuan(value: float) -> str:
    return f"{value:.4f} 元"


def _percent_number(value: float) -> str:
    return f"{value:.4f}%"


def _rate(value: float) -> str:
    return f"{value * 100:.4f}%"


def format_report(result: dict) -> str:
    underlying = result["underlying"]
    selected = result["selected"]
    dividends = result["dividends"]
    gap = result["call_model"] - result["call_market"]
    relative = gap / result["call_market"]
    candidate_text = "、".join(
        f"{item['month']}（到期 {item['expiry'].isoformat()}）" for item in result["candidates"]
    )
    dividend_flag = "是" if dividends["dividend_before_expiry"] else "否"
    lines = [
        "上证50ETF认购期权 Black-Scholes-Merton 入门定价",
        "=" * 52,
        f"估值日: {underlying['quote_date'].isoformat()} {underlying['quote_time']}",
        f"标的: {UNDERLYING} {underlying['name']}",
        f"标的最新价 S: {_yuan(underlying['spot'])}",
        f"候选到期月份: {candidate_text}",
        f"选用合约: {result['call']['合约交易代码']}",
        f"到期日: {selected['expiry'].isoformat()}",
        f"剩余自然日: {selected['calendar_days']}",
        f"期限 T: {result['year_fraction']:.6f} 年",
        f"行权价 K: {_yuan(result['strike'])}",
        f"认购市场价: {_yuan(result['call_market'])}",
        f"同执行价认沽: {result['put']['合约交易代码']}",
        f"认沽市场价: {_yuan(result['put_market'])}",
        "",
        "模型输入",
        f"历史波动率 σ: {_rate(result['volatility'])}（最近 {HISTORY_WINDOW} 个交易日对数收益，前复权）",
        f"Shibor 期限: {result['tenor_name']}",
        f"Shibor 报价日: {result['shibor']['date'].isoformat()}",
        f"Shibor 报价: {_percent_number(result['shibor']['percent_quote'])}",
        f"连续复利 r: {_rate(result['rate'])}",
        f"最近一次除息日: {dividends['last_ex_date'].isoformat()}",
        f"最近一次现金分红: {_yuan(dividends['cash_dividend'])}",
        f"推测下一次除息日: {dividends['next_ex_date'].isoformat()}",
        f"到期前是否分红: {dividend_flag}",
        f"连续股息率 q: {_rate(dividends['dividend_yield'])}",
        "",
        "定价结果",
        f"认购理论价: {_yuan(result['call_model'])}",
        f"认沽理论价: {_yuan(result['put_model'])}",
        f"认购 Delta: {result['call_delta']:.4f}",
        f"模型平价残差: {result['model_parity_gap']:.6f} 元",
        "",
        "和市场价比较",
        f"认购理论价 - 市场价: {gap:+.4f} 元/份",
        f"相对市场价: {relative:+.2%}",
        f"折合一张合约: {gap * CONTRACT_MULTIPLIER:+.2f} 元（乘数 {CONTRACT_MULTIPLIER}）",
        f"市场平价残差: {result['market_parity_gap']:+.4f} 元/份",
        "",
        "结论",
        build_conclusion(
            result["call_model"],
            result["call_market"],
            result["volatility"],
            dividends["dividend_yield"],
            result["tenor_name"],
        ),
        "市场平价残差不为零，是因为比较用的是最新成交价，买卖报价之间还有价差。",
    ]
    return "\n".join(lines)


def verify_textbook_examples() -> None:
    """打印并检查两组教科书数字。q=0 与 q=0.02 各一组，用来确认公式写对了。"""
    examples = (
        {
            "q": 0.0,
            "call": 10.4506,
            "put": 5.5735,
            "delta": 0.6368,
        },
        {
            "q": 0.02,
            "call": 9.2270,
            "put": 6.3301,
            "delta": 0.5869,
        },
    )
    print("公式自检（S=100, K=100, T=1, r=0.05, σ=0.20）")
    for example in examples:
        args = (100.0, 100.0, 1.0, 0.05, 0.20, example["q"])
        call = bsm_price(*args, "call")
        put = bsm_price(*args, "put")
        delta = bsm_delta(*args, "call")
        if round(call, 4) != example["call"] or round(put, 4) != example["put"]:
            raise AssertionError(f"教科书价格自检失败: q={example['q']}")
        if round(delta, 4) != example["delta"]:
            raise AssertionError(f"教科书 Delta 自检失败: q={example['q']}")
        print(
            f"  q={example['q']:.2f}  认购 {call:.4f}  认沽 {put:.4f}  "
            f"认购 Delta {delta:.4f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="用 Black-Scholes-Merton 给 50ETF 认购期权做入门定价")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只做教科书公式自检，不访问行情",
    )
    args = parser.parse_args(argv)
    verify_textbook_examples()
    if args.check_only:
        return
    print()
    print(format_report(price_live_contract()))


if __name__ == "__main__":
    main()
