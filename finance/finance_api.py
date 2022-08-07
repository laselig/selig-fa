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
from datetime import datetime, timedelta
import gc
from dateutil.relativedelta import relativedelta, MO, TU

sns.set_style("darkgrid")


def json_to_file(my_dict, out_file):
    """
    :param my_dict: python `dict` object
    :param out_file: where to save the dict object
    :return:
    """
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(my_dict, f, ensure_ascii=False, indent=4)
    print(f"wrote json to: {out_file}")


def request_and_download_stocks(ticker, save_path):

    # this thread works on a single ticker
    df = pd.read_parquet("/users/lselig/finance/finance/.data/fundamental_analysis_neglog.parquet")
    df_for_ticker = df[df.symbol == ticker]
    del df
    gc.collect()

    while not Path(save_path).is_file():
        prices = []
        # for all financial statements for the ticker
        for i in range(df_for_ticker.shape[0]):

            dt = df_for_ticker.iloc[i].date
            # dt = parser.parse(df_for_ticker.iloc[i].date)
            # relative delta doesn't jump back to previous monday if current date is a monday
            is_monday = dt.weekday() == 0
            offset = 1 if not is_monday else 2
            prev_monday = dt + relativedelta(weekday=TU(-1 * offset))
            next_monday = dt + relativedelta(weekday=TU(1 * offset))

            str_date = datetime.strftime(dt, "%Y-%m-%d")
            prev_date = datetime.strftime(prev_monday, "%Y-%m-%d")
            next_date = datetime.strftime(next_monday, "%Y-%m-%d")

            relevant_dates = [str_date, prev_date, next_date]
            labels = ["on", "prev", "next"]
            tmp = []
            for j, rd in enumerate(relevant_dates):
                n_fails = 0
                code = 429
                while code == 429 and n_fails <= 1000:
                    try:
                        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={rd}&to={rd}&apikey={API_KEY}"
                        response = urlopen(url)
                        code = response.code
                        if code != 200:
                            print(code, ticker)
                    except:
                        n_fails += 1
                        continue

                ret = json.loads(response.read().decode("utf-8"))
                daily_prices = pd.DataFrame().from_dict(ret)
                if daily_prices.shape[0] != 0:
                    daily_prices = pd.json_normalize(daily_prices.historical)
                    old_cols = daily_prices.columns
                    new_cols = [f"{x}_{labels[j]}" for x in old_cols]
                    name_map = dict(zip(old_cols, new_cols))
                    daily_prices = daily_prices.rename(columns=name_map)
                    tmp.append(daily_prices)

            if len(tmp) != 0:
                combined = pd.concat(tmp, axis=1)
                combined["symbol"] = ticker
                prices.append(combined)
            else:
                print("Empty request for a single statement")

        if len(prices) != 0:
            prices = pd.concat(prices, ignore_index=True)
            prices.to_parquet(save_path, index=False)
            return
        else:
            print(f"Empty df_for_ticker for all statements for a stock: {ticker}")
            pd.DataFrame().to_parquet(save_path, index=False)
            return


def request_and_download_financials(url, save_path):
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
            # response = urlopen(url)
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
    # response = urlopen(url, cafile=certifi.where())
    response = urlopen(url)
    ret = json.loads(response.read().decode("utf-8"))
    return ret


def get_stock_data(folder_name, df):
    unique_tickers = df.symbol.unique()
    base_path_out = f"{DATA_DIR}/{folder_name}"

    if not os.path.isdir(base_path_out):
        os.makedirs(base_path_out)
    save_paths = [f"{base_path_out}/{x}.parquet" for x in unique_tickers]
    request_save_map = dict(zip(unique_tickers, save_paths))
    # for debugging multithreading
    # request_save_map = {"AAPL": request_save_map["AAPL"]}

    tic = timeit.default_timer()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_download = {
            executor.submit(request_and_download_stocks, url, request_save_map[url]): url for url in request_save_map
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
            executor.submit(request_and_download_financials, url, request_save_map[url]): url
            for url in request_save_map
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


def run(pull_income_statements, pull_balance_sheets, pull_cash_flow, period, pull_stock_data):
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
    if pull_stock_data:
        folder_name = "hist_prices"
        all_financials_df = pd.read_parquet("/users/lselig/finance/finance/.data/fundamental_analysis_neglog.parquet")
        get_stock_data(folder_name, all_financials_df)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    API_KEY = os.environ.get("API_KEY")
    # pull all fundamental financials on a quarterly basis for publicly traded companies
    run(
        pull_income_statements=False,
        pull_cash_flow=False,
        pull_balance_sheets=False,
        period="quarter",
        pull_stock_data=True,
    )

    #     print(rd, daily_prices.date, daily_prices.open, daily_prices.high, daily_prices.close)
    # print("\n\n")
