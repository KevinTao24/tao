import akshare as ak
import pandas as pd

from .utils import analyze_max_up_down, get_avg_rates, get_esti_conti_days

stock_a = [
    "sh000001",
    "601318",
    "601398",
    "002142",
    "002593",
    "001979",
    "600410",
    "601825",
    "002129",
    "002261",
    "002312",
    "002512",
    "601800",
    "601985",
    "601727",
    "601857",
    "600941",
    "601099",
    "002272",
    "600895",
    "601166",
    "601319",
    "000627",
    "600520",
    "600839",
    "000595",
    "002423",
    "600837",
    "600696",
    "603259",
    "601162",
    "000158",
    "600611",
    "000062",
    "002584",
]

stock_etf = [
    "159501",
    "159529",
    "513350",
    "159792",
    "513180",
    "513050",
    "513730",
    "159561",
    "513520",
    "159934",
    "588000",
    "512200",
    "512660",
    "512170",
    "159869",
    "159892",
    "515790",
    "512690",
    "513970",
    "512480",
    "159599",
    "512880",
    "513090",
]

columns = [
    "股票名称",
    "涨幅",
    "连涨",
    "连涨幅",
    "30均涨",
    "30均跌",
    "30连涨",
    "30连跌",
    "50连涨",
    "50连跌",
    "80连涨",
    "100连涨",
    "200连涨",
    "历史连涨",
    "80连跌",
    "100连跌",
    "200连跌",
    "历史连跌",
    "均涨幅",
    "均跌幅",
    "60均涨幅",
    "60均跌幅",
    "5涨幅",
    "8涨幅",
    "16涨幅",
    "30涨幅",
    "50涨幅",
    "100涨幅",
    "200涨幅",
    "日期",
    "股票代码",
]


def analyze_stocks(stocks: list = stock_a) -> pd.DataFrame:
    df = pd.DataFrame(columns=columns)

    for stock in stocks:
        stock_name = ""
        if stock == "sh000001":
            stock = "000001"
            stock_name = "上证指数"
            stock_df = ak.index_zh_a_hist(symbol=stock, period="daily")
        else:
            stock_name = ak.stock_individual_info_em(symbol=stock).iloc[1].value
            stock_df = ak.stock_zh_a_hist(symbol=stock, period="daily", adjust="qfq")

        _analyze_stock(df, stock_df, stock_name, stock)

    df.set_index("股票名称", inplace=True)
    df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    return df


def analyze_etfs(etfs: list = stock_etf) -> pd.DataFrame:
    df = pd.DataFrame(columns=columns)
    etf_df = ak.fund_etf_spot_em()

    for etf in etfs:
        stock_name = etf_df[etf_df["代码"] == etf].iloc[-1].名称
        stock_df = ak.fund_etf_hist_em(symbol=etf, period="daily", adjust="qfq")
        _analyze_stock(df, stock_df, stock_name, etf)

    df.set_index("股票名称", inplace=True)
    df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
    return df


def _analyze_stock(
    df: pd.DataFrame, stock_df: pd.DataFrame, stock_name: str, stock: str
):
    # 连涨, 连涨幅
    conti_day = get_esti_conti_days(
        stock_df["涨跌幅"].tolist()[-30:-1], stock_df["涨跌幅"].tolist()[-1]
    )
    conti_rate = round(
        (
            (
                stock_df[-(abs(conti_day) + 1) :].iloc[-1]["收盘"]
                - stock_df[-(abs(conti_day) + 1) :].iloc[0]["收盘"]
            )
            / stock_df[-(abs(conti_day) + 1) :].iloc[0]["收盘"]
        )
        * 100,
        2,
    )

    # 均涨幅, 30均涨幅, 60均涨幅, 均跌幅, 30均跌幅, 60均跌幅
    avg_rate_up, avg_rate_down = get_avg_rates(stock_df["涨跌幅"].tolist()[-800:])
    avg_up_30, avg_down_30 = get_avg_rates(stock_df["涨跌幅"].tolist()[-30:])
    avg_up_60, avg_down_60 = get_avg_rates(stock_df["涨跌幅"].tolist()[-60:])

    # 5涨幅, 8涨幅, 16涨幅, 30涨幅, 50涨幅, 100涨幅, 200涨幅
    rate_x = []
    for i in [5, 8, 16, 30, 50, 100, 200]:
        rate = round(
            (
                (
                    stock_df[-(i + 1) :].iloc[-1]["收盘"]
                    - stock_df[-(i + 1) :].iloc[0]["收盘"]
                )
                / stock_df[-(i + 1) :].iloc[0]["收盘"]
            )
            * 100,
            2,
        )
        rate_x.append(rate)

    # 30/50/80/100/200/历史连涨连跌
    max_up_x, max_down_x = [], []
    max_up_cnt_x, max_down_cnt_x = [], []
    up_rate_x, down_rate_x = [], []
    for i in [30, 50, 80, 100, 200, 800]:
        max_up, max_down, max_up_count, max_down_count, max_up_rate, max_down_rate = (
            analyze_max_up_down(stock_df["涨跌幅"].tolist()[-i:], stock_df)
        )

        max_up_x.append(max_up)
        max_down_x.append(max_down)
        max_up_cnt_x.append(max_up_count)
        max_down_cnt_x.append(max_down_count)
        up_rate_x.append(max_up_rate)
        down_rate_x.append(max_down_rate)

    df.loc[len(df), columns] = [
        stock_name,
        stock_df.iloc[-1].涨跌幅,
        conti_day,
        conti_rate,
        avg_up_30,
        avg_down_30,
        str(max_up_x[0]) + "/" + str(max_up_cnt_x[0]) + "/" + str(up_rate_x[0]),
        str(max_down_x[0]) + "/" + str(max_down_cnt_x[0]) + "/" + str(down_rate_x[0]),
        str(max_up_x[1]) + "/" + str(max_up_cnt_x[1]) + "/" + str(up_rate_x[1]),
        str(max_down_x[1]) + "/" + str(max_down_cnt_x[1]) + "/" + str(down_rate_x[1]),
        max_up_x[2],
        max_up_x[3],
        max_up_x[4],
        max_up_x[5],
        max_down_x[2],
        max_down_x[3],
        max_down_x[4],
        max_down_x[5],
        avg_rate_up,
        avg_rate_down,
        avg_up_60,
        avg_down_60,
        rate_x[0],
        rate_x[1],
        rate_x[2],
        rate_x[3],
        rate_x[4],
        rate_x[5],
        rate_x[6],
        stock_df.iloc[-1].日期,
        stock,
    ]


def analyze_filter_stocks() -> pd.DataFrame:
    return pd.DataFrame()
