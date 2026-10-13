"""沪深300波动状态备忘：两状态马尔可夫机制转换模型的参考答案。

实际问题
    指数增强小组开风控例会时，需要回答三件事：当前沪深300更接近
    「平稳」还是「动荡」；这两种状态一般持续多久；若动荡概率已经
    升高，把风险预算降到一半会怎样。本脚本给出这份备忘。

模型（入门设定）
    对数收益率 r_t = μ_{S_t} + ε_t，ε_t ~ N(0, σ_{S_t}^2)。
    S_t 只有两个状态，明天的状态只取决于今天。估计完成后，
    方差较小的状态称为平稳，方差较大的状态称为动荡。

用法
    python markov_regime_csi300.py
    python markov_regime_csi300.py --start 2016-01-01 --end 2025-12-31
    python markov_regime_csi300.py --end latest --threshold 0.7
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

TRADING_DAYS = 252
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SINA_SYMBOL = {"000300": "sh000300"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="沪深300两状态马尔可夫机制转换备忘")
    parser.add_argument("--symbol", default="000300", help="指数代码，默认沪深300")
    parser.add_argument("--start", default="2016-01-01", help="样本开始日期 YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default="2025-12-31",
        help="样本结束日期 YYYY-MM-DD；填 latest 则取到运行当日",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="前一日动荡滤波概率达到该阈值时，当日仓位降为 50%%",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="备忘、图和概率表的保存目录",
    )
    args = parser.parse_args(argv)
    if not 0 < args.threshold < 1:
        parser.error("--threshold 需要在 0 和 1 之间")
    if args.end == "latest":
        args.end = pd.Timestamp.today().strftime("%Y-%m-%d")
    return args


def fetch_csi300(start: str, end: str, symbol: str = "000300") -> pd.DataFrame:
    """用 akshare 下载指数日线。东方财富失败时改用新浪。"""
    errors = []
    try:
        frame = _fetch_eastmoney(symbol, start, end)
        source = "akshare.index_zh_a_hist（东方财富）"
    except Exception as exc:  # 网络或接口字段变化时还有下一条路
        errors.append(f"index_zh_a_hist 失败：{exc}")
        try:
            frame = _fetch_sina(symbol, start, end)
            source = "akshare.stock_zh_index_daily（新浪财经）"
        except Exception as fallback_exc:
            errors.append(f"stock_zh_index_daily 失败：{fallback_exc}")
            raise RuntimeError("akshare 的两个指数日线接口都不可用。\n" + "\n".join(errors)) from fallback_exc
    frame.attrs["source"] = source
    print(f"数据接口：{source}")
    print(f"指数 {symbol}，{frame['date'].min():%Y-%m-%d} 至 {frame['date'].max():%Y-%m-%d}，共 {len(frame)} 个交易日。")
    return frame


def _fetch_eastmoney(symbol: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.index_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
    )
    return _clean_prices(raw, date_col="日期", close_col="收盘", start=start, end=end)


def _fetch_sina(symbol: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    sina_symbol = SINA_SYMBOL.get(symbol, symbol)
    raw = ak.stock_zh_index_daily(symbol=sina_symbol)
    return _clean_prices(raw, date_col="date", close_col="close", start=start, end=end)


def _clean_prices(raw: pd.DataFrame, date_col: str, close_col: str, start: str, end: str) -> pd.DataFrame:
    if date_col not in raw.columns or close_col not in raw.columns:
        raise ValueError(f"行情表缺少 {date_col} 或 {close_col}，实际列为 {list(raw.columns)}")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
        }
    )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)]
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    frame = frame[frame["close"] > 0].drop_duplicates("date", keep="last")
    if frame.empty:
        raise ValueError(f"{start} 至 {end} 没有可用的收盘价")
    return frame.reset_index(drop=True)


def add_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """对数收益率用于估计，简单收益率用于连乘收益。"""
    out = prices.set_index("date").sort_index()
    out["log_ret"] = np.log(out["close"]).diff()
    out["simple_ret"] = out["close"].pct_change()
    out = out.dropna(subset=["log_ret", "simple_ret"])
    if len(out) < 100:
        raise RuntimeError("有效收益率少于 100 个交易日，两状态模型不稳定。请放宽样本区间。")
    return out


def fit_two_state(log_ret: pd.Series):
    """每种状态有自己的均值和方差，状态之间按马尔可夫链切换。"""
    # 交易日索引含节假日空档，statsmodels 会提示无法推断频率。本案例不做预测，忽略即可。
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A date index has been provided")
        model = MarkovRegression(
            log_ret,
            k_regimes=2,
            trend="c",
            switching_variance=True,
        )
        result = model.fit(maxiter=200, disp=False)
    converged = getattr(result, "mle_retvals", {}).get("converged", True)
    if not converged:
        print("警告：优化器报告未收敛。可以把样本改长一些，或稍后重试。")
    return result


def name_regimes(params: pd.Series) -> dict[str, int]:
    """估计完成后再命名。波动小的叫平稳，波动大的叫动荡。"""
    variance = {0: float(params["sigma2[0]"]), 1: float(params["sigma2[1]"])}
    if np.isclose(variance[0], variance[1]):
        raise RuntimeError("两种状态的方差几乎相同，模型没有把行情分开。请更换样本区间后再估计。")
    calm = min(variance, key=variance.get)
    return {"平稳": calm, "动荡": 1 - calm}


def transition_matrix(params: pd.Series) -> pd.DataFrame:
    """行是今天的状态编号，列是明天的状态编号。"""
    stay_0 = float(params["p[0->0]"])
    to_0_from_1 = float(params["p[1->0]"])
    matrix = pd.DataFrame(
        [[stay_0, 1.0 - stay_0], [to_0_from_1, 1.0 - to_0_from_1]],
        index=[0, 1],
        columns=[0, 1],
    )
    return matrix


def expected_duration(stay_probability: float) -> float:
    """期望持续期 = 1 / (1 - 留在本状态的概率)，单位是交易日。"""
    if stay_probability >= 1:
        return float("inf")
    return 1.0 / (1.0 - stay_probability)


def state_probability(result, regime_id: int, kind: str) -> pd.Series:
    table = getattr(result, f"{kind}_marginal_probabilities").copy()
    table.columns = [int(column) for column in table.columns]
    return table[regime_id].astype(float)


def trailing_true_count(mask: pd.Series) -> int:
    """从序列末尾往前，连续为真的个数。"""
    count = 0
    for value in reversed(mask.astype(bool).tolist()):
        if not value:
            break
        count += 1
    return count


def position_from_filtered(filtered_turbulent: pd.Series, threshold: float) -> pd.Series:
    """用昨天的滤波概率决定今天的仓位，避免用到当天尚未走完的信息。

    概率达到阈值：仓位 50%。否则：仓位 100%。第一天没有昨日概率，仓位为 100%。
    """
    lagged = filtered_turbulent.shift(1)
    position = pd.Series(1.0, index=filtered_turbulent.index, dtype=float)
    position.loc[lagged >= threshold] = 0.5
    return position


def performance(simple_returns: pd.Series, position: pd.Series) -> dict[str, float | pd.Series]:
    """现金收益按 0 计算。最大回撤是负数，越接近 0 越浅。"""
    aligned = simple_returns.astype(float) * position.astype(float)
    wealth = (1.0 + aligned).cumprod()
    observations = len(aligned)
    return {
        "cumulative": float(wealth.iloc[-1] - 1.0),
        "ann_return": float(wealth.iloc[-1] ** (TRADING_DAYS / observations) - 1.0),
        "ann_vol": float(aligned.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "average_position": float(position.mean()),
        "wealth": wealth,
    }


def risk_clause(rule: dict, hold: dict) -> str:
    quieter = rule["ann_vol"] < hold["ann_vol"]
    shallower = rule["max_drawdown"] > hold["max_drawdown"]
    if quieter and shallower:
        risk = "年化波动更低，最大回撤更浅"
    elif quieter:
        risk = "年化波动更低，但最大回撤没有更浅"
    elif shallower:
        risk = "最大回撤更浅，但年化波动没有更低"
    else:
        risk = "年化波动和最大回撤都没有改善"
    if rule["ann_return"] < hold["ann_return"]:
        payoff = "年化收益低于一直满仓"
    else:
        payoff = "年化收益不低于一直满仓"
    return f"{risk}，同时{payoff}"


def annualized_mean(daily_mean: float) -> float:
    return daily_mean * TRADING_DAYS


def annualized_volatility(daily_variance: float) -> float:
    return float(np.sqrt(daily_variance * TRADING_DAYS))


def build_report(result, data: pd.DataFrame, threshold: float) -> dict:
    params = result.params
    pvalues = result.pvalues
    names = name_regimes(params)
    matrix = transition_matrix(params)
    durations = {
        label: expected_duration(float(matrix.loc[regime, regime]))
        for label, regime in names.items()
    }
    filtered = state_probability(result, names["动荡"], "filtered")
    smoothed = state_probability(result, names["动荡"], "smoothed")
    position = position_from_filtered(filtered, threshold)
    rule = performance(data["simple_ret"], position)
    hold = performance(data["simple_ret"], pd.Series(1.0, index=data.index))
    profiles = {}
    for label, regime in names.items():
        daily_mean = float(params[f"const[{regime}]"])
        daily_variance = float(params[f"sigma2[{regime}]"])
        profiles[label] = {
            "regime": regime,
            "daily_mean": daily_mean,
            "ann_return": annualized_mean(daily_mean),
            "ann_vol": annualized_volatility(daily_variance),
            "mean_pvalue": float(pvalues[f"const[{regime}]"]),
            "duration": durations[label],
        }
    latest = data.index[-1]
    filtered_now = float(filtered.loc[latest])
    current = "动荡" if filtered_now >= threshold else "平稳"
    in_current = filtered >= threshold if current == "动荡" else filtered < threshold
    return {
        "names": names,
        "matrix": matrix,
        "profiles": profiles,
        "filtered": filtered,
        "smoothed": smoothed,
        "position": position,
        "rule": rule,
        "hold": hold,
        "threshold": threshold,
        "current": current,
        "filtered_now": filtered_now,
        "smoothed_now": float(smoothed.loc[latest]),
        "turbulent_share": float((filtered >= threshold).mean()),
        "current_spell": trailing_true_count(in_current),
    }


def render_memo(data: pd.DataFrame, report: dict, source: str) -> str:
    start = data.index[0]
    end = data.index[-1]
    calm = report["profiles"]["平稳"]
    turbulent = report["profiles"]["动荡"]
    names = report["names"]
    matrix = report["matrix"]
    calm_id = names["平稳"]
    turbulent_id = names["动荡"]
    rule = report["rule"]
    hold = report["hold"]
    comparison = risk_clause(rule, hold)
    duration_now = report["profiles"][report["current"]]["duration"]
    conclusion = (
        f"截至{end:%Y-%m-%d}，动荡状态的滤波概率为{pct(report['filtered_now'])}。"
        f"当前更接近「{report['current']}」，这种状态的期望持续期约{duration_now:.0f}个交易日。"
        f"样本内在动荡概率升高时把仓位降到一半，{comparison}。"
        "该对照使用了全样本参数，只说明规则含义，不能当作实盘业绩。"
    )
    lines = [
        "=" * 62,
        "沪深300波动状态备忘",
        f"数据接口：{source}",
        f"收益率样本：{start:%Y-%m-%d} 至 {end:%Y-%m-%d}，共 {len(data)} 个交易日",
        "模型：两状态马尔可夫机制转换（每种状态有自己的均值和方差）",
        "命名规则：估计完成后，方差较小的状态称为平稳，较大的称为动荡",
        "=" * 62,
        "",
        "一、两种状态",
        _profile_line("平稳", calm),
        _profile_line("动荡", turbulent),
        "年化收益 = 日均对数收益 × 252；年化波动 = sqrt(日方差 × 252)。",
        "均值 p 值如果大于 0.05，不要把该状态说成显著的上涨或下跌行情。",
        f"动荡状态的年化波动大约是平稳状态的 {turbulent['ann_vol'] / calm['ann_vol']:.1f} 倍。",
        "",
        "二、转移概率",
        (
            f"今天平稳：明天仍平稳 {pct(matrix.loc[calm_id, calm_id])}，"
            f"明天转入动荡 {pct(matrix.loc[calm_id, turbulent_id])}。"
        ),
        (
            f"今天动荡：明天转入平稳 {pct(matrix.loc[turbulent_id, calm_id])}，"
            f"明天仍动荡 {pct(matrix.loc[turbulent_id, turbulent_id])}。"
        ),
        "期望持续期 = 1 / (1 - 留在本状态的概率)，单位是交易日。",
        "这是一段状态的平均总长度，不是从今天开始的倒计时。",
        "马尔可夫的含义是：明天的转移概率只看今天在哪个状态，与这段状态已经走了多久无关。",
        "",
        f"三、样本最后一天 {end:%Y-%m-%d}",
        f"动荡状态滤波概率：{pct(report['filtered_now'])}",
        f"动荡状态平滑概率：{pct(report['smoothed_now'])}",
        "最后一天两者相同，因为此时还没有未来数据。判断“当前”看这一对数字即可。",
        "历史复盘涂色用平滑概率；决定当时仓位只能用已经发生的滤波概率。",
        f"按阈值 {pct(report['threshold'], 0)} 判定，当前更接近「{report['current']}」。",
        f"这一判定已经连续保持了 {report['current_spell']} 个交易日。",
        "",
        "四、教学用风险预算规则（不是投资建议）",
        f"规则：前一交易日动荡滤波概率 ≥ {pct(report['threshold'], 0)} 时，当日仓位 50%，否则 100%。",
        "现金部分收益按 0 计算，不考虑交易成本。参数用全样本估计，所以这是样本内演示。",
        _performance_line("一直满仓", hold),
        _performance_line("降风险预算", rule),
        f"动荡滤波概率达到阈值的交易日占比：{pct(report['turbulent_share'])}。",
        "",
        "五、结论",
        conclusion,
        "=" * 62,
    ]
    return "\n".join(lines)


def _profile_line(label: str, profile: dict) -> str:
    return (
        f"{label}：年化收益 {pct(profile['ann_return'])}，"
        f"年化波动 {pct(profile['ann_vol'])}，"
        f"期望持续 {profile['duration']:.1f} 个交易日，"
        f"均值 p 值 {profile['mean_pvalue']:.3f}。"
    )


def _performance_line(label: str, stats: dict) -> str:
    return (
        f"{label}：累计收益 {pct(stats['cumulative'])}，"
        f"年化收益 {pct(stats['ann_return'])}，"
        f"年化波动 {pct(stats['ann_vol'])}，"
        f"最大回撤 {pct(stats['max_drawdown'])}，"
        f"平均仓位 {pct(stats['average_position'])}。"
    )


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def save_probability_table(data: pd.DataFrame, report: dict, path: Path) -> None:
    table = pd.DataFrame(
        {
            "日期": data.index,
            "收盘价": data["close"].to_numpy(),
            "对数收益率": data["log_ret"].to_numpy(),
            "简单收益率": data["simple_ret"].to_numpy(),
            "动荡滤波概率": report["filtered"].to_numpy(),
            "动荡平滑概率": report["smoothed"].to_numpy(),
            "教学仓位": report["position"].to_numpy(),
        }
    )
    table.to_csv(path, index=False, encoding="utf-8-sig")


def plot_regimes(data: pd.DataFrame, report: dict, path: Path) -> None:
    configure_chinese_font()
    smoothed = report["smoothed"]
    filtered = report["filtered"]
    threshold = report["threshold"]
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")

    price_axis = axes[0]
    for left, right in contiguous_spans(smoothed >= 0.5):
        price_axis.axvspan(left, right, color="#f4a261", alpha=0.35, linewidth=0)
    price_line = price_axis.plot(data.index, data["close"], color="#1d3557", linewidth=1.0, label="收盘价")[0]
    price_axis.set_xlim(data.index[0], data.index[-1])
    price_axis.set_ylabel("收盘价")
    price_axis.set_title("沪深300：阴影是事后看属于动荡的交易日（平滑概率 ≥ 50%）")
    price_axis.legend(
        handles=[
            price_line,
            Patch(facecolor="#f4a261", alpha=0.35, label="动荡（平滑概率 ≥ 50%）"),
        ],
        loc="upper left",
    )
    price_axis.grid(True, axis="y", alpha=0.25)

    probability_axis = axes[1]
    probability_axis.plot(filtered.index, filtered.to_numpy(), color="#e76f51", linewidth=1.0, label="动荡滤波概率")
    probability_axis.axhline(threshold, color="#6d6875", linestyle="--", linewidth=1.0, label=f"阈值 {pct(threshold, 0)}")
    probability_axis.set_ylim(0, 1)
    probability_axis.set_ylabel("概率")
    probability_axis.set_xlabel("日期")
    probability_axis.set_title("当时能看到的动荡概率（滤波概率，决策使用前一天的值）")
    probability_axis.legend(loc="upper left")
    probability_axis.grid(True, axis="y", alpha=0.25)

    figure.savefig(path, dpi=140)
    plt.close(figure)


def contiguous_spans(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    spans = []
    start = None
    previous = None
    for timestamp, flagged in mask.items():
        if flagged and start is None:
            start = timestamp
        if not flagged and start is not None:
            spans.append((start, _span_end(start, previous)))
            start = None
        previous = timestamp
    if start is not None:
        spans.append((start, _span_end(start, previous)))
    return spans


def _span_end(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    if end <= start:
        return start + pd.Timedelta(days=1)
    return end


def configure_chinese_font() -> None:
    candidates = [
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            break
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "PingFang SC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prices = fetch_csi300(args.start, args.end, args.symbol)
    data = add_returns(prices)
    result = fit_two_state(data["log_ret"])
    print()
    print(result.summary())
    print()
    report = build_report(result, data, args.threshold)
    memo = render_memo(data, report, prices.attrs.get("source", "akshare"))
    print(memo)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memo_path = output_dir / "备忘.txt"
    figure_path = output_dir / "波动状态.png"
    table_path = output_dir / "状态概率.csv"
    memo_path.write_text(memo + "\n", encoding="utf-8")
    save_probability_table(data, report, table_path)
    plot_regimes(data, report, figure_path)
    print(f"备忘已保存：{memo_path}")
    print(f"概率表已保存：{table_path}")
    print(f"图已保存：{figure_path}")


if __name__ == "__main__":
    main()
