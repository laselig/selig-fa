import certifi, time, json, timeit, os
import numpy as np
import concurrent.futures
from pathlib import Path
from dateutil import parser
import seaborn as sns
import pandas as pd
from urllib.request import urlopen
from alive_progress import alive_bar
from finance.constants import MAX_WORKERS, DATA_DIR
from dotenv import load_dotenv, find_dotenv
sns.set_style("darkgrid")

def json_to_file(my_dict, out_file):
    """
    :param my_dict: python `dict` object
    :param out_file: where to save the dict object
    :return:
    """
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(my_dict, f, ensure_ascii=False, indent=4)
    print(f"Wrote json to: {out_file}")


def request_and_download(url, save_path):
    """
    :param url: the api request
    :param save_path: the base location for where to save the data
    :return:
    """
    # spam the api at top speed until we get all of the data
    # code 429: we are currently timed out (hit api request limit)
    # also sometimes they have dead end apis so we can escape those by failing a single request >= n times (1000 seems okay)
    code = 429
    n_fails = 0
    while code == 429 and n_fails <= 1000:
        try:
            response = urlopen(url, cafile=certifi.where())
            code = response.code
        except:
            n_fails += 1
            continue
        tmp_dict = json.loads(response.read().decode("utf-8"))
        tmp_df = pd.DataFrame().from_dict(tmp_dict)
        if "date" in list(tmp_df):
            tmp_df["date"] = [parser.parse(x) for x in tmp_df.date]
            tmp_df = tmp_df.sort_values(by=["date"])
            tmp_df.to_parquet(save_path, index=False)
        else:
            return
    return


def make_single_api_request(url):
    """
    :param url: endpoint
    :return: response
    """
    response = urlopen(url, cafile=certifi.where())
    ret = json.loads(response.read().decode("utf-8"))
    return ret


def get_financial_statement_data(folder_name, base_api_url, period):
    """
    :param folder_name: base save dir for all of the downloads
    :param base_api_url: base api end point
    :param period: "quarter" for quarterly reports: https://site.financialmodelingprep.com/developer/docs
    :return:
    """
    tickers = make_single_api_request(
        f"https://financialmodelingprep.com/api/v3/financial-statement-symbol-lists?apikey={API_KEY}"
    )
    relevant_tickers = [x for x in tickers if x.count(".") == 0]
    request_save_map = {}
    base_path_out = f"{DATA_DIR}/{folder_name}"
    if not os.path.isdir(base_path_out):
        os.makedirs(base_path_out)
    for t in relevant_tickers:
        f_name = f"{t}.parquet"
        out_path = Path(base_path_out) / f_name
        url = f"{base_api_url}/{t}?period={period}&apikey={API_KEY}"
        request_save_map[url] = out_path

    tic = timeit.default_timer()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_download = {
            executor.submit(request_and_download, url, request_save_map[url]): url for url in request_save_map
        }
        with alive_bar(
            len(future_download),
            bar="smooth",
            title=f"Pulling {folder_name}..",
        ) as progress_bar:
            for i, future in enumerate(concurrent.futures.as_completed(future_download)):
                download = future_download[future]
                try:
                    _ = future.result()
                except Exception as exc:
                    print(f"%{download} generated an exception: {exc}\nSkipped and moving on")
                progress_bar()
    toc = timeit.default_timer()
    print(f"Pulling {folder_name} took : {np.round( toc - tic, 2 )}s")


def run(pull_income_statements, pull_balance_sheets, pull_cash_flow, period):
    if pull_income_statements:
        folder_name = "income_statements"
        base_api_url = "https://financialmodelingprep.com/api/v3/income-statement"
        get_financial_statement_data(folder_name, base_api_url, period=period)
    if pull_cash_flow:
        folder_name = "cash_flows"
        base_api_url = "https://financialmodelingprep.com/api/v3/cash-flow-statement"
        get_financial_statement_data(folder_name, base_api_url, period=period)
    if pull_balance_sheets:
        folder_name = "balance_sheets"
        base_api_url = "https://financialmodelingprep.com/api/v3/balance-sheet-statement"
        get_financial_statement_data(folder_name, base_api_url, period=period)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    API_KEY = os.environ.get("API_KEY")
    # pull all fundamental financials on a quarterly basis for publicly traded companies
    run(
        pull_income_statements=True,
        pull_cash_flow=True,
        pull_balance_sheets=True,
        period="quarter",
    )
# my_df = my_df.sort_values(by = ["date"])
# print(my_df)
#
# from_ = "2002-01-01"
# to = "2022-07-30"
#
# daily_prices = pd.DataFrame().from_dict(make_api_request(f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={from_}&to={to}&apikey={API_KEY}"))
# daily_prices = pd.json_normalize(daily_prices.historical)
# daily_prices["ticker"] = [ticker] * daily_prices.shape[0]
# print(daily_prices.shape)
# # daily_prices = daily_prices.sort_values(by = ["date"])
#
# fig, axs = plt.subplots(6, 1, figsize = (15, 9), sharex = True)
# print(my_df.date)
# axs[0].plot([parser.parse(x) for x in my_df.date], my_df.revenue)
# axs[0].set_ylabel("revenue")
#
# axs[1].plot([parser.parse(x) for x in my_df.date], my_df.grossProfit)
# axs[1].set_ylabel("gross profit")
#
# axs[2].plot([parser.parse(x) for x in my_df.date], my_df.operatingExpenses)
# axs[2].set_ylabel("operating expenses")
# for i in range(3):
#     axs[i].set_xticklabels( axs[i].get_xticks(), rotation = 45 )
#     axs[i].set_yscale('log')
#
# axs[3].plot([parser.parse(x) for x in my_df.date], my_df.depreciationAndAmortization)
# axs[3].set_ylabel("depreciation/amortization")
#
# axs[4].plot([parser.parse(x) for x in daily_prices.date], daily_prices.open, label = "open")
# axs[4].plot([parser.parse(x) for x in daily_prices.date], daily_prices.close, label = "close")
# axs[4].plot([parser.parse(x) for x in daily_prices.date], daily_prices.high, label = "high")
#
# axs[4].legend()
# fig.suptitle(f'{ticker}')
# plt.show()
# print(f'Found {len(result)} quarterly income statements')
# for r in result:
#     out_path = f"{base_path_out}/{ticker}_{period}_{r['date']}_incomestatement.json"
#     print(r)
#     json_to_file(r, out_path)
# dfs = []
# for i, file in enumerate(glob.glob(f"{base_path_out}/*.json")):
#     f = open(file)
#     try:
#         raw_data_dict = json.load(f)
#     except:
#         continue
#     print(i, raw_data_dict)
#     data = pd.DataFrame.from_dict([raw_data_dict])
#     dfs.append(data)
#     print(data)
# print(pd.concat(dfs))
# df_path = f'{ticker}_income_statments.parquet'
