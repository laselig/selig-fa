import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, random
from pathlib import Path
from alive_progress import alive_bar
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # this is the project root
MAX_WORKERS = os.cpu_count()
DATA_DIR = f"{ROOT_DIR}/.data"
PLOTS_DIR = f"{ROOT_DIR}/.plots"

np.random.seed(0)
sns.set_style("darkgrid")


def folders_to_dfs(loop_over):
    # create a df from each of the 3 folder types
    # loop_over = ["income_statements", "balance_sheets", "cash_flows"]
    non_floats = ["symbol", "date", "period", "year"]
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
                    if(df.shape[0] != 0):
                        df = df.drop(columns = ["dcf", "dcfDiff"], axis = 1)
                    for col in list(df):
                        if(type_ != "profile" and col in non_floats):
                            df[col] = df[col].astype("float64")
                    try:
                        df["othertotalStockholdersEquity"] = df.othertotalStockholdersEquity.astype("float64")
                    except:
                        pass
                    tmp.append(df)
        df = pd.concat(tmp)
        df.to_parquet(f"{DATA_DIR}/{type_}.parquet", index=False)


def neglog(x):
    return np.sign(x) * np.log10(np.abs(x) + 1)

def combine_evs_finratios():
    evs = pd.read_parquet(f"{DATA_DIR}/evs.parquet")
    ratios = pd.read_parquet(f"{DATA_DIR}/ratios.parquet")
    insider_trades = pd.read_parquet(f"{DATA_DIR}/insider_trades.parquet")
    profiles = pd.read_parquet(f"{DATA_DIR}/profile.parquet")
    keep_me = ["exchangeShortName", "industry", "sector", "country",
               "fullTimeEmployees", "isEtf", "isActivelyTrading", "isFund", "symbol"]
    profiles = profiles[keep_me]

    print(evs.head())
    print(ratios.head())

    evs_ratios = pd.merge(evs, ratios, on = ["symbol", "date"])
    evs_ratios["year"] = pd.DatetimeIndex(evs_ratios["date"]).year
    my_dict = {"Q1" : 1.0,
               "Q2": 2.0,
               "Q3": 3.0,
               "Q4": 4.0}
    evs_ratios = evs_ratios.replace({"period": my_dict})
    evs_ratios = evs_ratios[evs_ratios.period != ""]
    evs_ratios["period"] = evs_ratios.period.astype("float64")
    evs_ratios = evs_ratios.rename(columns = {"period": "quarter"})


    evs_ratios = pd.merge(evs_ratios, insider_trades, on = ["symbol", "year", "quarter"])
    final = pd.merge(evs_ratios, profiles, on = ["symbol"])
    final.to_parquet(f"{DATA_DIR}/evs_ratios.parquet")

    return

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
    df_tmp["current_ratio"] = df_tmp.totalCurrentAssets / df_tmp.totalCurrentLiabilities
    df_tmp["share_number"] = df_tmp.commonStock + df_tmp.preferredStock
    market_cap = (df_tmp.share_number) * df_tmp.open_on
    enterprise_value = market_cap - df_tmp.cashAndCashEquivalents + df_tmp.totalDebt

    ratios = ["pe_ratio", "ps_ratio", "pcf_ratio", "pfcf_ratio",
            "pb_ratio", "ev_sales", "ev_over_ebitda", "ev_to_cash_flow",
            "earnings_yield", "free_cash_flow_yield", "debt_to_equity",
            "debt_to_assets", "current_ratio"]

    price = df_tmp.open_prev
    df_tmp["pe_ratio"] = price / (df_tmp.netIncome_x / df_tmp.share_number)
    df_tmp["ps_ratio"] = price / (df_tmp.revenue / df_tmp.share_number)
    df_tmp["pcf_ratio"] = price / (df_tmp.operatingCashFlow / df_tmp.share_number)
    df_tmp["pfcf_ratio"] = market_cap / df_tmp.freeCashFlow
    df_tmp["pb_ratio"] = price  / (df_tmp.totalStockholdersEquity / df_tmp.share_number)
    df_tmp["ev_sales"] = enterprise_value / df_tmp.revenue
    df_tmp["ev_over_ebitda"] = enterprise_value / df_tmp.ebitda
    df_tmp["ev_to_cash_flow"] = enterprise_value / df_tmp.operatingCashFlow
    df_tmp["earnings_yield"] = (df_tmp.netIncome_x / df_tmp.share_number) / price
    df_tmp["free_cash_flow_yield"] = df_tmp.freeCashFlow / market_cap
    df_tmp["debt_to_equity"] = df_tmp.longTermDebt / df_tmp.totalStockholdersEquity
    df_tmp["debt_to_assets"] = df_tmp.longTermDebt / df_tmp.totalAssets
    df_tmp["current_ratio"] = df_tmp.totalCurrentAssets / df_tmp.totalCurrentLiabilities

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
            if feat in skip_transform or feat in ratios:
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
    # loop_over = ["ratios"]
    # loop_over = ["evs"]
    # loop_over = ["insider_trades"]
    # folders_to_dfs(loop_over)
    # folders_to_dfs(["profile"])
    combine_evs_finratios()
    # print("Converting dfs to one big df")
    # dfs_to_master_df(do_plots=True, do_summary=True)


if __name__ == "__main__":
    run()
