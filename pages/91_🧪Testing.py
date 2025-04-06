import pandas as pd
import streamlit as st

from apps.stockers.analyze import analyze_etfs, analyze_stocks

int_cols = [
    "连涨",
    "80连涨",
    "80连跌",
    "100连涨",
    "100连跌",
    "200连涨",
    "200连跌",
    "历史连涨",
    "历史连跌",
]

str_cols = [
    "30连涨",
    "30连跌",
    "50连涨",
    "50连跌",
]

float_cols = [
    "涨幅",
    "连涨幅",
    "均涨幅",
    "均跌幅",
    "30均涨",
    "30均跌",
    "60均涨幅",
    "60均跌幅",
    "5涨幅",
    "8涨幅",
    "16涨幅",
    "30涨幅",
    "50涨幅",
    "100涨幅",
    "200涨幅",
]


def highlight_pos_neg(val):
    if pd.isna(val):
        return ""
    try:
        if isinstance(val, str):
            if "/" in val:
                if "-" in val:
                    return "color: green; font-weight: bold"
                else:
                    return f"color: red; font-weight: bold"

        val_float = float(val)
        if val_float > 0:
            return f"color: red; font-weight: bold"
        elif val_float < 0:
            return "color: green; font-weight: bold"
        else:
            return "color: black; font-weight: bold"
    except ValueError:
        return ""


def main():
    st.set_page_config("Testing", page_icon="🚀")
    st.title("🧪 Testing")
    st.caption("🚀 A Streamlit testing powered by AIGC")

    st.subheader("Analyze Stock")
    with st.spinner("Fetching data..."):
        df = analyze_stocks()

        if df.empty:
            st.error("No data found. Please check the ticker or date range.")
        else:
            df_show = df.copy()
            styled_df = (
                df_show.style.map(
                    highlight_pos_neg, subset=int_cols + str_cols + float_cols
                )
                .format("{:d}", subset=int_cols)
                .format("{}", subset=str_cols)
                .format("{:.2f}", subset=float_cols)
            )
            st.write(styled_df)

    st.subheader("Analyze ETF")
    with st.spinner("Fetching data..."):
        etf_df = analyze_etfs()

        if etf_df.empty:
            st.error("No data found. Please check the ticker or date range.")
        else:
            etf_df_show = etf_df.copy()
            etf_styled_df = (
                etf_df_show.style.map(
                    highlight_pos_neg, subset=int_cols + str_cols + float_cols
                )
                .format("{:d}", subset=int_cols)
                .format("{}", subset=str_cols)
                .format("{:.2f}", subset=float_cols)
            )
            st.write(etf_styled_df)


if __name__ == "__main__":
    main()
