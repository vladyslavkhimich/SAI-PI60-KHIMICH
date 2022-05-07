import numpy as np
import datetime
import json
import matplotlib.pyplot as plt
#from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_yahoo
from sklearn import covariance, cluster

# !!! quotes_historical_yahoo_ochl is now deprecated

# Вхідний файл із символічними позначеннями компаній
input_file = 'company_symbol_mapping.json'
