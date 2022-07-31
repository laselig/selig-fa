# Fundamental Analysis Tools
This repository contains python and julia scripts for getting, cleaning and analyzing fundamental financials. Strong focus on value investing rather than technical analysis
### Covers:
- NYSE
- NASDAQ
- AMEX
- EURONEXT
- TSX
- Indexes
- ETFs/Mutual Funds
- Forex
- Crypto
---  
## Requirements
| Name                | Description              |
| ------------------- | ------------------------ |
| Python Version      | 3.8.5 <= version < 3.9.0 |
| Package Manager     | Poetry                   |
| Starter API Key     | https://site.financialmodelingprep.com/developer/docs/ |

---
## Setup
1. Open up terminal
2. Type ```git clone https://github.com/laselig/selig-fa.git``` in terminal
3. Type ```cd selig-fa``` in terminal
4. Type ```poetry install``` in terminal
5. Type ```source .venv/bin/activate``` in terminal to setup your environment
6. Setup your constants file located at: ```selig-fa/constants.py```
7. Type ```python finance_api.py``` in terminal to pull the financial data (one time setup) 
8. Type ```python finance_wrangling.py``` in terminal to clean and process the data (one time setup)
9. Explore the ```finance_ds.ipynb``` notebook to understand how to interact with the data 


## Goals
1. Create a robust and clean dataset composed of stock fundamental financial data
2. Explore the data to identify opportunities for investment
    - Clustering analysis (unsupervised learning) based on quarterly balance sheets, cash flow sheets and income statements
    - Value each company based on (1) using regression models (supervised learning) 
        - In other words, can a companies stock price be predicted from the most basic fundamentals? 

## TODO
1. Hook up data pipelines for: 
    - insider trading volumes
    - stock prices 1 wk before and after fundamental financial documents were made publicly available
2. Build a UI/web app to more easily dive into the data    
