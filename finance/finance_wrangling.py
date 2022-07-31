"""
CONFIDENTIAL
__________________
2022 Happy Health Incorporated
All Rights Reserved.
NOTICE:  All information contained herein is, and remains
the property of Happy Health Incorporated and its suppliers,
if any.  The intellectual and technical concepts contained
herein are proprietary to Happy Health Incorporated
and its suppliers and may be covered by U.S. and Foreign Patents,
patents in process, and are protected by trade secret or copyright law.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Happy Health Incorporated.
Authors: Lucas Selig <lucas@happy.ai>
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, random
from pathlib import Path
from finance.constants import DATA_DIR, PLOTS_DIR

np.random.seed(0)
sns.set_style("darkgrid")


def folders_to_dfs():
    # create a df from each of the 3 folder types
    loop_over = ["income_statements", "balance_sheets", "cash_flows"]
    for type_ in loop_over:
        indv_files = glob.glob(f"{DATA_DIR}/{type_}/*")
        tmp = []
        for i, x in enumerate(indv_files):
            # skip tickers with . in the name
            if x.count(".") >= 3:
                continue
            else:
                df = pd.read_parquet(x)
                try:
                    df["othertotalStockholdersEquity"] = df.othertotalStockholdersEquity.astype("float64")
                except:
                    continue
                tmp.append(df)
        df = pd.concat(tmp)
        df.to_parquet(f"{DATA_DIR}/{type_}.parquet", index=False)


def dfs_to_master_df(do_plots, do_summary):
    """
    Combines the income statement, cash flow and balance sheet dfs into a merged and cleaned df
    Saves 2 outputs: raw combined and feature engineered combined parquets
    :param do_plots: save some distrubtion plots for each of the features
    :param do_summary: print out summary info about the merged df
    :return:
    """
    cash_flow = pd.read_parquet(f"{DATA_DIR}/cash_flow.parquet")
    cash_flow = cash_flow[cash_flow.reportedCurrency == "USD"]
    cash_flow = cash_flow.drop(columns=["link", "finalLink", "date", "reportedCurrency", "cik", "fillingDate"])

    income_statement = pd.read_parquet(f"{DATA_DIR}/income_statement.parquet")
    income_statement = income_statement[income_statement.reportedCurrency == "USD"]
    income_statement = income_statement.drop(
        columns=["link", "finalLink", "date", "reportedCurrency", "cik", "fillingDate"]
    )
    balance_sheet = pd.read_parquet(f"{DATA_DIR}/balance_sheet.parquet")
    balance_sheet = balance_sheet[balance_sheet.reportedCurrency == "USD"]
    balance_sheet = balance_sheet.drop(columns=["link", "finalLink", "date", "reportedCurrency", "cik", "fillingDate"])
    df_tmp = pd.merge(income_statement, cash_flow, on=["symbol", "period", "calendarYear"])
    df_tmp = pd.merge(df_tmp, balance_sheet, on=["symbol", "period", "calendarYear"])
    # save raw combined
    df_tmp.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    df_tmp = df_tmp.drop(columns = ["acceptedDate_x", "acceptedDate_y"])
    df_tmp.to_parquet(
        f"{DATA_DIR}/fundamental_analysis.parquet",
        index=False,
    )

    # starter feature engineering
    features = list(df_tmp)[3:]
    for feat in features:
        if("acceptedDate" in feat):
            continue
        transformed_feat = np.sign(df_tmp[feat]) * np.log10(np.abs(df_tmp[feat]) + 1)
        df_tmp[feat] = transformed_feat
        if(do_plots):
            sns.kdeplot(transformed_feat, fill = True)
            lower = np.nanmin(df_tmp[feat])
            upper = np.nanmax(df_tmp[feat])
            plt.suptitle(f"neglog transform: {True}--{feat}\n"
                         f"lower: {lower:e}\n"
                         f"upper: {upper:e}")
            plt.savefig(f"{PLOTS_DIR}/{feat}_dist.png", dpi = 500)
            plt.close()
    df_tmp.to_parquet(f"{DATA_DIR}/fundamental_analysis_neglog.parquet", index = False)
    if(do_summary):
        print(df_tmp.head())
        print(df_tmp.describe())
        print(df_tmp.info())
        print(df_tmp.columns)
        print(df_tmp["acceptedDate"])
