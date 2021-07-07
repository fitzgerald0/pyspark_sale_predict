# encoding: utf-8
# Author    : huangjisheng
# Datetime  : 2021/4/21 16:55
# Product   : PyCharm
# File      : describe_time_series.py

"""

use the spark.udf and statsmodels to test the behaviors of the cyclicity and weekly stationarity of any Large-Scale Time Series data

"""

import datetime
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark import SparkConf
from pyspark.sql.functions import pandas_udf, PandasUDFType

spark = SparkSession. \
    Builder(). \
    config("spark.sql.crossJoin.enabled", "true"). \
    config("spark.sql.execution.arrow.enabled", "false"). \
    enableHiveSupport(). \
    getOrCreate()

#base statistic feature

item_sale_sql = """
select 
shop_id,
sku_id,
avg(sale_qty) as sale_avg,
stddev_samp(sale_qty) as sale_std,
stddev_samp(sale_qty)/avg(sale_qty) as cv_coef,
skewness(sale_qty) as sale_skew,
kurtosis(sale_qty) as sale_kurt,
min(dt) as start_sale,
max(dt) as end_sale,
count(dt) as sale_days,
floor(datediff(max(dt), min(dt)))+1 as sale_span,
(floor(datediff(max(dt), min(dt)))+1)  - count(dt) as intermit_day,
count(dt)/(floor(datediff(max(dt), min(dt)))+1) as sale_ratio,
count(dt)*(((skewness(sale_qty)*skewness(sale_qty))+((kurtosis(sale_qty)-3)*(kurtosis(sale_qty)-3))/4)/6) as item_jb
from temp.dataset_feature
where sale_qty>0
group by shop_id,sku_id"""


item_sale_sql = item_sale_sql.format(train_end_date=train_end_date)
item_sale_stats = sc.sql(item_sale_sql)
item_sale_stats.createOrReplaceTempView('item_sale_stats')

schema_adfuller = StructType([
    StructField("shop_id", StringType()),
    StructField("sku_id", StringType()),
    StructField("adfuller", IntegerType())])

@pandas_udf(schema_adfuller, functionType=PandasUDFType.GROUPED_MAP)
def adf_test(df):
    df.sort_values(by=['date'],ascending=[True],inplace=True)
    adf_values=adfuller(df['qty'],autolag='AIC')[1]
    is_adf_func = lambda x: 1 if x < 0.05 else 0
    is_adf = is_adf_func(adf_values)
    result=pd.DataFrame({'store_id':df['store_id'].iloc[0],'is_adf':[is_adf]})
    return result

schema_cycle = StructType([
    StructField("shop_id", StringType()),
    StructField("sku_id", StringType()),
    StructField("cycle", IntegerType())])


@pandas_udf(schema_cycle, functionType=PandasUDFType.GROUPED_MAP)
def cycle_test(df,check_len=7):
    """
     check series the cycle
    :param df: data of series
    :param check_len: will check cycle length,eg weekly:7,month:30,year:12
    :return:int,1 is exit cycle,other is 0
    """
    assert check_len >2,('cycle length must bigger then 2')
    df = df.sort_values(by=['dt'], ascending=True)
    acf_values=acf(df['label'],nlags=df.shape[0]-1)
    loc_max_index=signal.argrelextrema(acf_values,comparator=np.greater,order=check_len//2)

    #7 is weekly cycle if month data series can choice 12
    #occur local max index in cycle for len, as be exit cycle
    cycle_check=[i for i in loc_max_index[0] if i%check_len==0]
    is_cycle=lambda  x : 1 if cycle_check else 0
    cycle_result=is_cycle(cycle_check)
    result = pd.DataFrame({'shop_id':df['shop_id'].iloc[0],'sku_id':df['sku_id'].iloc[0], 'cycle': [cycle_result]})
    return result


def long_trend(data):
    """
    :param data: DataFrame of one series
    :return: DataFrame of data index(key) info and regression coefficient as long_trend
    """
    slop=lambda x: linregress(list(range(1,data.shape[0]+1)),data.label.values)
    trend=slop(data)[0]
    return pd.DataFrame({'shop_id':df['shop_id'].iloc[0],'sku_id':df['sku_id'].iloc[0], 'long_trend': [trend]})

def model_input():
    data_sql = """select shop_id,sku_id,dt,label from temp.dataset_feature"""
    data = spark.sql(data_sql)
    return data

if __name__ == '__main__':
    data=model_input()
    data =data.na.fill(0)
    item_adfuller = data.groupby(['shop_id','sku_id']).apply(adf_test)
    item_adfuller.createOrReplaceTempView('item_adfuller')
    item_cycle = data.groupby(['shop_id','sku_id']).apply(cycle_test)
    try:
        feature_df_tab.write.mode("append").format('hive').saveAsTable('temp.item_adfuller_cycle_table')
    except:
        item_cycle.createOrReplaceTempView('item_cycle')
        spark.sql("""drop table if exists temp.item_adfuller_cycle_table""")
        spark.sql("""create table temp.item_adfuller_cycle_table as 
        select a.shop_id,a.sku_id,a.cycle,b.adfuller
        from item_cycle a 
        left join item_adfuller b
        on a.shop_id=b.shop_id and a.sku_id=b.sku_id""")
