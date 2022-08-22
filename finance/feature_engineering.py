import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, random
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from feature_engine.outliers import Winsorizer
from feature_engine.discretisation import DecisionTreeDiscretiser, EqualFrequencyDiscretiser
from feature_engine.encoding import OneHotEncoder
from dateutil import parser
import datetime

np.random.seed(0)
sns.set_style("darkgrid")

df = pd.read_parquet("/Users/lselig/selig-fa/finance/.data/evs_ratios.parquet")
# df["year"] = pd.DatetimeIndex(df["date"]).year
# df = df[df.symbol.isin(["AAPL", "GOOGL", "MSFT", "GME", "A", "QQQ", "AMZN", "TSLA"])]
# df = df[df.year >= 2015]
yrs_maturity = 3

df = df[(df.stockPrice >= 4) & (df.stockPrice <= 1000)]
remove_me = []
for col in list(df):
    num_na = df[col].isna().sum().sum()
    if(num_na > 30000):
        remove_me.append(col)

remove_me.append("fullTimeEmployees")
df = df.drop(columns = remove_me)
df = df.dropna()
df = df[df.sector != ""]
df = df[df.ipoDate != ""]
df["daysSinceIPO"] = [(parser.parse(x) - datetime.datetime.now()).days * -1 for x in df.ipoDate]
df = df[df.daysSinceIPO >= yrs_maturity * 365]
df = df[(df.isFund == False) & (df.isEtf == False) & (df.country == "US") &
        ((df.exchangeShortName == "NASDAQ") | (df.exchangeShortName == "NYSE"))]




threshold = 21
value_counts = df["industry"].value_counts() # Specific column
print(value_counts.to_string())
to_remove = value_counts[value_counts <= threshold].index
df["industry"] = df["industry"].replace(to_remove, np.nan)
df = df.dropna()
# df = df[df.industry != "Biotechnology"]
meta_cols = ["stockPrice", "symbol", "quarter", "cik",
             "isEtf", "isActivelyTrading", "isFund", "country", "ipoDate"]
features = df


def neglog(x):
    return (np.sign(x) * np.log10(np.abs(x) + 1))
def get_outlier_idxs(feature, feature_name):

    use_mad = True
    use_quantile = False
    ignore_pct = 0.01

    if(use_mad):
        magic_c = 0.6745
        cutoff_value = 3.0
        mad = np.nanmedian(np.abs(feature - np.nanmedian(feature)))
        mi_feature = (magic_c * (feature - np.nanmedian(feature))) / mad
        outliers = np.where(np.abs(mi_feature) >= cutoff_value)[0]
        print(f"{feature_name} Found {len(outliers)} outliers out of {feature.shape[0]} -- {len(outliers) / feature.shape[0] * 100: .2f}%")


    elif(use_quantile):
        lower = np.nanquantile(feature, ignore_pct)
        upper = np.nanquantile(feature, 1 - ignore_pct)
        outliers = np.where((feature >= upper) | (feature <= lower))[0]
        print(f"{feature_name} Found {len(outliers)} outliers out of {feature.shape[0]} -- {len(outliers) / feature.shape[0] * 100: .2f}%")

    return outliers

discretisize_me = ["addTotalDebt", "payoutRatio", "buySellRatio", "totalBought", "totalSold"]
skip_transform_me = ["stockPrice", "exchangeShortName", "industry", "sector", "date", "year"]
skip_transform_me += meta_cols
for f in features:
    if(f in skip_transform_me):
        continue
    raw = features[f]
    transformed = neglog(raw)
    fig, axs = plt.subplots(2, 4, figsize = (12, 9))
    axs = axs.flatten()
    axs[0].boxplot(raw)
    axs[0].set_title("Raw")

    outlier_remover = Winsorizer(capping_method = "iqr",
                                 tail = "both")
    raw_iqr = outlier_remover.fit_transform(raw.values.reshape(-1, 1))
    axs[1].boxplot(raw_iqr)
    axs[1].set_title("Raw + IQR")

    axs[2].boxplot(transformed)
    axs[2].set_title("Log(Raw)")
    outlier_remover = Winsorizer(capping_method = "gaussian",
                                 tail = "both")
    log_gaussian = outlier_remover.fit_transform(transformed.values.reshape(-1, 1))
    # print(log_gaussian)


    # outlier_idx = get_outlier_idxs(raw, f)
    # raw_log_gaussian = raw.values[~outlier_idx]
    axs[3].boxplot(log_gaussian)
    axs[3].set_title("Log(Raw) + Gaussian removal")
    axs[4].hist(raw, bins = 40)
    axs[5].hist(raw_iqr, bins = 40)
    axs[6].hist(transformed, bins = 40)
    axs[7].hist(log_gaussian, bins = 40)
    axs[2].boxplot(transformed)
    axs[2].set_title("neglog")
    transformed2 = neglog(transformed)
    axs[3].boxplot(transformed2)
    axs[3].set_title("neglog(neglog)")
    fig.suptitle(f)
    plt.tight_layout()
    # plt.show()
    plt.close()
    do_discretisize = True
    if(f in discretisize_me and do_discretisize):
        bin_me = EqualFrequencyDiscretiser()
        binned = bin_me.fit_transform(log_gaussian)
        features[f] = binned.values
    else:
        features[f] = log_gaussian.values
for f in features:
    if(f in skip_transform_me):
        continue
    print(f)
    plt.hist2d(features[f], np.log10(features["stockPrice"]), bins = 300)
    corrlog = np.corrcoef(features[f].values, np.log10(features["stockPrice"].values))[0, 1]
    corr = np.corrcoef(features[f].values, features["stockPrice"].values)[0, 1]
    plt.title(f"{f}\n{corr = }\n{corrlog = }")
    plt.savefig(f"/Users/lselig/selig-fa/finance/.plots/corr/{f}.png", dpi = 200)
    plt.close()





encoder = OneHotEncoder(variables=["exchangeShortName", "industry", "sector"])
# fit the encoder
res = encoder.fit_transform(features)

res.to_parquet("/Users/lselig/selig-fa/finance/.data/evs_ratios_preprocessed.parquet", index = False)
print("wrote preprocessed df to file")

