import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, random
from pathlib import Path
from finance.constants import DATA_DIR, PLOTS_DIR
from alive_progress import alive_bar

np.random.seed(0)
sns.set_style("darkgrid")


def folders_to_dfs(loop_over):
    # create a df from each of the 3 folder types
    # loop_over = ["income_statements", "balance_sheets", "cash_flows"]
    for type_ in loop_over:
        indv_files = glob.glob(f"{DATA_DIR}/{type_}/*")
        tmp = []
        with alive_bar(
            len(indv_files),
            bar="smooth",
            title=f"Combining {type_} dfs..",
        ) as progress_bar:

            for i, x in enumerate(indv_files):
                # skip tickers with . in the name
                progress_bar()
                if x.count(".") >= 3:
                    continue
                else:
                    df = pd.read_parquet(x)
                    try:
                        df["othertotalStockholdersEquity"] = df.othertotalStockholdersEquity.astype("float64")
                    except:
                        pass
                    tmp.append(df)
        df = pd.concat(tmp)
        df.to_parquet(f"{DATA_DIR}/{type_}.parquet", index=False)


def neglog(x):
    return np.sign(x) * np.log10(np.abs(x) + 1)


def dfs_to_master_df(do_plots, do_summary):
    """
    Combines the income statement, cash flow and balance sheet dfs into a merged and cleaned df
    Saves 2 outputs: raw combined and feature engineered combined parquets
    :param do_plots: save some distrubtion plots for each of the features
    :param do_summary: print out summary info about the merged df
    :return:
    """
    cash_flow = pd.read_parquet(f"{DATA_DIR}/cash_flows.parquet")
    cash_flow = cash_flow[cash_flow.reportedCurrency == "USD"]
    cash_flow = cash_flow.drop(columns=["link", "finalLink", "acceptedDate", "reportedCurrency", "cik", "fillingDate"])

    income_statement = pd.read_parquet(f"{DATA_DIR}/income_statements.parquet")
    income_statement = income_statement[income_statement.reportedCurrency == "USD"]
    income_statement = income_statement.drop(
        columns=["link", "finalLink", "acceptedDate", "reportedCurrency", "cik", "fillingDate"]
    )
    balance_sheet = pd.read_parquet(f"{DATA_DIR}/balance_sheets.parquet")
    balance_sheet = balance_sheet[balance_sheet.reportedCurrency == "USD"]
    balance_sheet = balance_sheet.drop(
        columns=["link", "finalLink", "acceptedDate", "reportedCurrency", "cik", "fillingDate"]
    )

    hist_prices = pd.read_parquet(f"{DATA_DIR}/hist_prices.parquet")
    hist_prices["date_on"] = pd.to_datetime(hist_prices.date_on)
    # hist_prices = hist_prices.astype({'date_on': 'int32'}).dtypes
    hist_prices = hist_prices.rename(columns={"date_on": "date"})
    hist_prices = hist_prices[
        ["symbol", "date", "open_on", "volume_on", "vwap_on", "open_next", "open_prev", "volume_prev", "vwap_prev"]
    ]
    # want to predict pct increase of stock given it's financial info on day of financial statements
    hist_prices["wk_curr_to_next_pct_inc"] = hist_prices.open_next / hist_prices.open_on
    hist_prices["wk_prev_to_curr_pct_inc"] = hist_prices.open_on / hist_prices.open_prev

    df_tmp = pd.merge(income_statement, cash_flow, on=["symbol", "period", "calendarYear"])
    df_tmp = pd.merge(df_tmp, balance_sheet, on=["symbol", "period", "calendarYear"])
    df_tmp = pd.merge(df_tmp, hist_prices, on=["symbol", "date"])

    # save raw combined
    df_tmp.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    df_tmp = df_tmp.drop(columns=["date_x", "date_y"])
    df_tmp = df_tmp.dropna()
    print("num na: ", df_tmp.isna().sum().sum())
    df_tmp.to_parquet(
        f"{DATA_DIR}/fundamental_analysis.parquet",
        index=False,
    )

    # starter feature engineering
    skip_transform = [
        "symbol",
        "date",
        "open_on",
        "open_prev",
        "vwap_on",
        "open_next",
        "vwap_next",
        "wk_curr_to_next_pct_inc",
        "wk_prev_to_curr_pct_inc",
    ]
    features = list(df_tmp)
    # double transform ratios
    if not Path(f"{PLOTS_DIR}").is_dir():
        os.makedirs(f"{PLOTS_DIR}")

    for feat in features:
        print("Working on", feat)
        if "date" in feat or "symbol" in feat or "calendarYear" in feat or "period" in feat:
            continue
        elif feat in skip_transform:
            transformed_feat = df_tmp[feat]
            transformed = False
        else:
            transformed = True
            transformed_feat = neglog(df_tmp[feat])

        if do_plots:
            sns.kdeplot(transformed_feat, fill=True)
            lower = np.nanmin(df_tmp[feat])
            upper = np.nanmax(df_tmp[feat])
            plt.suptitle(f"neglog transform: {transformed}--{feat}\n" f"lower: {lower:e}\n" f"upper: {upper:e}")
            plt.savefig(f"{PLOTS_DIR}/{feat}_dist.png", dpi=500)
            if feat in skip_transform:
                plt.show()
            plt.close()
        df_tmp[feat] = transformed_feat
    df_tmp.to_parquet(f"{DATA_DIR}/fundamental_analysis_neglog.parquet", index=False)

    if do_summary:
        print(df_tmp.head())
        print(df_tmp.describe())
        print(df_tmp.info())
        print(df_tmp.columns)


def run():
    # loop_over = ["income_statements", "balance_sheets", "cash_flows"]
    # loop_over = ["hist_prices"]
    # folders_to_dfs(loop_over)
    print("Converting dfs to one big df")
    dfs_to_master_df(do_plots=True, do_summary=True)


if __name__ == "__main__":
    run()
