# encoding: utf-8
# Author    : huangjisheng
# Datetime  : 2021/4/20 14:39
# Product   : PyCharm
# File      : simple_model.py

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
spark = SparkSession. \
    Builder(). \
    config("spark.sql.crossJoin.enabled", "true"). \
    config("spark.sql.execution.arrow.enabled", "false"). \
    enableHiveSupport(). \
    getOrCreate()


#naive forecast method
#use windows functions of lag in Spark.SQL
naive_cycle_sql="""
select
shop_number,
sku_number,
sale_qty,
dt,
lag(sale_qty,7,0) over(partition by shop_number,sku_number order by dt) as naive_predict
from app.forecast_dataset
"""
spark.sql(naive_cycle_sql)
spark.stop()


# moving average method
def moving_average(data, window_size,n_step):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec[-n_step:]


# Simple Exponential Smoothing method
def es_model(series,alpha=0.3):
    """
    smooth series data
    ----------
    :param X:input series of pd.DataFrame one columns data
    :param alpha: smooth
    :return Simple Exponential Smoothing forecast
    """
    X=series.values
    s = [X[0]]
    for i in range(1, len(X)):
        temp =alpha * X[i] + (1 - alpha) * s[-1]
        s.append(temp)
    return s

