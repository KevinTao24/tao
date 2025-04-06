import akshare as ak
import pandas as pd
import backtrader as bt


ts_code = "601398"
df = ak.stock_zh_a_hist(symbol=ts_code, period="daily", adjust="qfq")


# 获取 A 股数据（AKShare）
def get_stock_data(stock_code, start_date, end_date):
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        },
        inplace=True,
    )
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df["openinterest"] = 0  # Backtrader 需要这个字段
    return df[["open", "high", "low", "close", "volume", "openinterest"]]


# **定投策略**
class DCA_Strategy(bt.Strategy):
    params = (("investment", 1000),)  # 每月定投金额

    def __init__(self):
        self.last_buy_date = None  # 记录上次买入日期

    def next(self):
        dt = self.datas[0].datetime.date(0)
        if self.last_buy_date is None or dt.month != self.last_buy_date.month:
            cash = self.broker.get_cash()
            price = self.datas[0].open[0]  # 当天开盘价
            num_shares = self.params.investment / price  # 计算可以买多少股
            if cash >= self.params.investment:
                self.buy(size=num_shares)
                self.last_buy_date = dt  # 更新买入日期


# **运行回测**
def run_backtest(stock_code, start_date, end_date, investment=1000):
    # 获取数据
    data = get_stock_data(stock_code, start_date, end_date)
    datafeed = bt.feeds.PandasData(dataname=data)

    # 初始化 Backtrader
    cerebro = bt.Cerebro()
    cerebro.addstrategy(DCA_Strategy, investment=investment)
    cerebro.adddata(datafeed)
    cerebro.broker.set_cash(20000)  # 初始资金 1 万
    cerebro.broker.setcommission(commission=0.001)  # 手续费 0.1%

    # 运行回测
    print(f"初始资金: {cerebro.broker.getvalue()} 元")
    cerebro.run()
    print(f"最终资金: {cerebro.broker.getvalue()} 元")


# **测试策略**
run_backtest("000533", "20240221", "20250309", investment=1000)  # 000001 平安银行


