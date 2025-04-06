from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def get_avg_rates(rates: List[float]) -> Tuple[float, float]:
    """Get the average values of rates up and down"""

    rates = np.array(rates)
    rate_up = np.mean(rates[rates > 0])
    rate_down = np.mean(rates[rates < 0])
    avg_rate_up = round(rate_up, 2) if not np.isnan(rate_up) else 0
    avg_rate_down = round(rate_down, 2) if not np.isnan(rate_down) else 0

    return avg_rate_up, avg_rate_down


def analyze_max_up_down(
    rates: List[float], df: pd.DataFrame
) -> Tuple[int, int, int, int, float, float]:
    """Analyze the maximum number of consecutive days of up and down by DP method."""

    n = len(rates)
    if n == 0:
        return 0, 0

    up_dp = [0] * n
    down_dp = [0] * n

    for i in range(1, n):
        if rates[i] > 0:
            up_dp[i] = up_dp[i - 1] + 1
        if rates[i] < 0:
            down_dp[i] = down_dp[i - 1] + 1

    max_up_dp = max(up_dp)
    max_down_dp = -max(down_dp)

    max_up_index = [i for i, val in enumerate(up_dp) if val == max_up_dp]
    max_down_index = [i for i, val in enumerate(down_dp) if val == -max_down_dp]

    tmp_up_rate = []
    for i in max_up_index:
        tmp = -n + i + 1
        if tmp == 0:
            tmp_df = df[-n + i - max_up_dp :].copy()
        else:
            tmp_df = df[-n + i - max_up_dp : -n + i + 1].copy()

        tmp_rate = round(
            (
                (tmp_df.iloc[-1]["收盘"] - tmp_df.iloc[0]["收盘"])
                / tmp_df.iloc[0]["收盘"]
            )
            * 100,
            2,
        )
        tmp_up_rate.append(tmp_rate)

    tmp_down_rate = []
    for i in max_down_index:
        tmp = -n + i + 1
        if tmp == 0:
            tmp_df = df[-n + i + max_down_dp :].copy()
        else:
            tmp_df = df[-n + i + max_down_dp : -n + i + 1].copy()

        tmp_rate = round(
            (
                (tmp_df.iloc[-1]["收盘"] - tmp_df.iloc[0]["收盘"])
                / tmp_df.iloc[0]["收盘"]
            )
            * 100,
            2,
        )
        tmp_down_rate.append(tmp_rate)

    return (
        max_up_dp,
        max_down_dp,
        up_dp.count(max_up_dp),
        down_dp.count(-max_down_dp),
        max(tmp_up_rate),
        min(tmp_down_rate),
    )


def get_max_up_down_days(rates: List[float]) -> Tuple[int, int]:
    """Get the maximum number of consecutive days of up and down by DP method."""

    n = len(rates)
    if n == 0:
        return 0, 0

    up_dp = [0] * n
    down_dp = [0] * n

    for i in range(1, n):
        if rates[i] > 0:
            up_dp[i] = up_dp[i - 1] + 1
        if rates[i] < 0:
            down_dp[i] = down_dp[i - 1] + 1

    return max(up_dp), -max(down_dp)


def get_max_up_down_days_pros(rates: List[float]) -> Tuple[int, int, Dict, Dict]:
    """Get the maximum number of consecutive days of up and down by DP method."""

    up_dp, down_dp = get_max_up_down_days(rates)
    up_num_counts = Counter(up_dp)
    down_num_counts = Counter(down_dp)
    up_counts = [(num, count) for num, count in up_num_counts.items() if num > 0]
    down_counts = [(num, count) for num, count in down_num_counts.items() if num > 0]

    up_pros = []
    for i in range(len(up_counts)):
        if i == 0:
            up_pros.append(0)
        else:
            prev_y = up_counts[i - 1][1]
            curr_y = up_counts[i][1]
            pro = round(100 * curr_y / prev_y, 2)
            up_pros.append(pro)

    up_pros_dict = {}
    if len(up_pros) > 2:
        up_pros_dict = {
            "pro_2": f"{up_pros[1]}%/{up_counts[0][1]}/{up_counts[1][1]}",
            "pro_3": f"{up_pros[2]}%/{up_counts[1][1]}/{up_counts[2][1]}",
        }

    down_pros = []
    for i in range(len(down_counts)):
        if i == 0:
            down_pros.append(0)
        else:
            prev_y = down_counts[i - 1][1]
            curr_y = down_counts[i][1]
            pro = round(100 * curr_y / prev_y, 2)
            down_pros.append(pro)

    down_pros_dict = {}
    if len(down_pros) > 2:
        down_pros_dict = {
            "pro_2": f"{down_pros[1]}%/{down_counts[0][1]}/{down_counts[1][1]}",
            "pro_3": f"{down_pros[2]}%/{down_counts[1][1]}/{down_counts[2][1]}",
        }

    return max(up_dp), -max(down_dp), up_pros_dict, down_pros_dict


def get_topn_days_stats(
    rates: List[float], days: int, topn: int
) -> Tuple[List[float], float, float, float]:
    """Get the TopN, Maximum, Minimum, Average values of `days` days"""

    max_index = len(rates) - days
    if max_index <= 0:
        return [], 0.0, 0.0, 0.0

    values = [round(sum(rates[i : i + days]), 2) for i in range(max_index + 1)]

    max_value = round(max(values), 2)
    min_value = round(min(values), 2)
    avg_value = round(np.mean(values), 2)
    topn_up_value = sorted(values, reverse=True)[:topn]
    topn_down_value = sorted(values, reverse=False)[:topn]

    return topn_up_value, topn_down_value, max_value, min_value, avg_value


def get_topn_conti_days_stats(
    rates: List[float],
    days: int,
    topn: int,
    is_up: bool,
) -> Tuple[List[float], float, float, float]:
    """Get the TopN, Maximum, Minimum, Average values of continue up or down for `days` days"""

    max_index = len(rates) - days
    if max_index <= 0:
        return [], 0.0, 0.0, 0.0

    condition = (
        (lambda r: all(rate > 0 for rate in r))
        if is_up
        else (lambda r: all(rate < 0 for rate in r))
    )
    values = [
        round(sum(rates[i : i + days]), 2)
        for i in range(max_index + 1)
        if condition(rates[i : i + days])
    ]

    max_value = round(max(values), 2) if len(values) != 0 else 0
    min_value = round(min(values), 2) if len(values) != 0 else 0
    avg_value = round(np.mean(values), 2) if len(values) != 0 else 0
    topn_value = sorted(values, reverse=is_up)[:topn]

    return topn_value, max_value, min_value, avg_value


def get_esti_conti_days(rates: List[float], rzzl: float) -> int:
    """Get the estimated number of continue up or down days."""

    if rzzl == 0:
        return 0

    conti_days = 1
    while (rzzl > 0 and rates[-conti_days] > 0) or (
        rzzl < 0 and rates[-conti_days] < 0
    ):
        conti_days += 1

    return conti_days if rzzl > 0 else -conti_days


def get_indicators(df: pd.DataFrame) -> Dict:
    ####### technical analysis indicator ######
    # ind1:mean 近5|10|15|20日涨幅均值，近5|10|15|20日跌幅均值，历史涨幅均值，历史跌幅均值
    # ind2:max 近5|10|15|20日涨幅最大值，近5|10|15|20日跌幅最大值，历史涨幅最大值，历史跌幅最大值
    # ind3:min 近5|10|15|20日涨幅最小值，近5|10|15|20日跌幅最小值，历史涨幅最小值，历史跌幅最小值
    # ind4:days 近10|15|20日最大连涨天数、最大连跌天数、最多连涨天数、最多连跌天数，历史最大连涨天数、历史最大连跌天数、历史最多连涨天数、历史最多连跌天数
    # ind5:topn 近20日Top5涨幅值、跌幅值，近50日Top10涨幅值、跌幅值，历史Top10涨幅值、跌幅值
    #############

    rates = list(df["rate"])
    indicators = {"mean": {}, "max": {}, "min": {}, "days": {}, "topn": {}}

    # ind1:mean 近5|10|15|20日涨幅均值，近5|10|15|20日跌幅均值，历史涨幅均值，历史跌幅均值
    for day in [5, 10, 15, 20]:
        indicators["mean"][f"up_mean_{day}"] = round(
            df[df["rate"] > 0]
            .rolling(day)
            .apply(lambda x: np.mean(x))
            .iloc[-1]["rate"],
            2,
        )
        indicators["mean"][f"down_mean_{day}"] = round(
            df[df["rate"] < 0]
            .rolling(day)
            .apply(lambda x: np.mean(x))
            .iloc[-1]["rate"],
            2,
        )

    indicators["mean"]["up_mean_all"] = round(
        float(df[df["rate"] > 0].apply(lambda x: np.mean(x))), 2
    )
    indicators["mean"]["down_mean_all"] = round(
        float(df[df["rate"] < 0].apply(lambda x: np.mean(x))), 2
    )

    # ind2:max 近5|10|15|20日涨幅最大值，近5|10|15|20日跌幅最大值，历史涨幅最大值，历史跌幅最大值
    # ind3:min 近5|10|15|20日涨幅最小值，近5|10|15|20日跌幅最小值，历史涨幅最小值，历史跌幅最小值
    for day in [10, 30, 60, 120]:
        indicators["mean"][f"up_mean_{day}"] = round(
            df[df["rate"] > 0]
            .rolling(day)
            .apply(lambda x: np.mean(x))
            .iloc[-1]["rate"],
            2,
        )
        indicators["mean"][f"down_mean_{day}"] = round(
            df[df["rate"] < 0]
            .rolling(day)
            .apply(lambda x: np.mean(x))
            .iloc[-1]["rate"],
            2,
        )

        indicators["max"][f"up_max_{day}"] = round(
            df[df["rate"] > 0].rolling(day).apply(lambda x: np.max(x)).iloc[-1]["rate"],
            2,
        )
        indicators["max"][f"down_max_{day}"] = round(
            df[df["rate"] < 0].rolling(day).apply(lambda x: np.min(x)).iloc[-1]["rate"],
            2,
        )

        indicators["min"][f"up_min_{day}"] = round(
            df[df["rate"] > 0].rolling(day).apply(lambda x: np.min(x)).iloc[-1]["rate"],
            2,
        )
        indicators["min"][f"down_min_{day}"] = round(
            df[df["rate"] < 0].rolling(day).apply(lambda x: np.max(x)).iloc[-1]["rate"],
            2,
        )

    indicators["max"]["up_max_all"] = round(
        float(df[df["rate"] > 0].apply(lambda x: np.max(x))), 2
    )
    indicators["max"]["down_max_all"] = round(
        float(df[df["rate"] < 0].apply(lambda x: np.min(x))), 2
    )
    indicators["min"]["up_min_all"] = round(
        float(df[df["rate"] > 0].apply(lambda x: np.min(x))), 2
    )
    indicators["min"]["down_min_all"] = round(
        float(df[df["rate"] < 0].apply(lambda x: np.max(x))), 2
    )

    # ind4:days 近10|15|20日最大连涨天数、最大连跌天数、最多连涨天数、最多连跌天数，历史最大连涨天数、历史最大连跌天数、历史最多连涨天数、历史最多连跌天数
    for day in [30, 60, 120]:
        (
            indicators["days"][f"up_max_conti_{day}"],
            indicators["days"][f"down_max_conti_{day}"],
            _,
            _,
        ) = get_max_up_down_days(rates[-day:])

    (
        indicators["days"][f"up_max_conti_all"],
        indicators["days"][f"down_max_conti_all"],
        indicators["days"][f"up_most_conti_all"],
        indicators["days"][f"down_most_conti_all"],
    ) = get_max_up_down_days(rates)

    # ind5:topn 近20日Top5涨幅值、跌幅值，近50日Top10涨幅值、跌幅值，历史Top10涨幅值、跌幅值
    for day, n in [(20, 5), (50, 10)]:
        df_day = df.iloc[-day:]
        indicators["topn"][f"up_top{n}_{day}"] = sorted(
            list(df_day[df_day["rate"] > 0]["rate"]), reverse=True
        )[:n]
        indicators["topn"][f"down_top{n}_{day}"] = sorted(
            list(df_day[df_day["rate"] < 0]["rate"])
        )[:n]

    indicators["topn"]["up_top10_all"] = sorted(
        list(df[df["rate"] > 0]["rate"]), reverse=True
    )[:10]
    indicators["topn"]["down_top10_all"] = sorted(list(df[df["rate"] < 0]["rate"]))[:10]

    indicators["topn"]["up_top10_all"] = [
        round(x, 2) for x in indicators["topn"]["up_top10_all"]
    ]
    indicators["topn"]["down_top10_all"] = [
        round(x, 2) for x in indicators["topn"]["down_top10_all"]
    ]

    return indicators
