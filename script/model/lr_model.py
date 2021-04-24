# encoding: utf-8
# Author    : huangjisheng
# Datetime  : 2021/4/16 13:39
# Product   : PyCharm
# File      : lr_model.py

import sys
import datetime
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder

"""
1.init spark
2.read dataset from  hive table 
3.transitions feature into vector
4.split train_set/predict_set
5.random split train_set into train and val
6.read tuning param from hive table,put into model
7.train/fit model 
8.inference
9.store the predict result as TempView into hive table
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

spark = SparkSession. \
    Builder(). \
    config("spark.sql.crossJoin.enabled", "true"). \
    config("spark.sql.execution.arrow.enabled", "false"). \
    enableHiveSupport(). \
    getOrCreate()


class linear_predict(object):
    def __init__(self,data,importance_feature,reg, inter, elastic):
        self.data=data
        self.importance_feature=importance_feature
        self.reg=reg
        self.inter=inter
        self.elastic=elastic

    def prediction(self):
        reg, inter, elastic=self.reg,self.inter,self.elastic
        df=self.data
        inputCols = self.importance_feature
        #feature must be not null
        df = df.na.fill(0)
        #transitions feature into vector
        feature_vector = VectorAssembler(inputCols=inputCols, outputCol="features")
        output = feature_vector.transform(df)
        features_label = output.select("shop_number", "item_number", "dt", "features", "label")
        #split train_set/predict_set
        train_set =features_label.where(features_label['dt'] <='2020-08-28')
        #random split train_set into train and val
        train_data, val_data = train_set.randomSplit([0.8, 0.2])
        pred_data=features_label.where(features_label['dt']>'2020-08-28').where(features_label['dt']<'2020-09-01')
        #read tuning param from hive table,put into model
        lr = LinearRegression(regParam=reg, fitIntercept=inter, elasticNetParam=elastic,solver="normal")
        model = lr.fit(train_data)
        print('{}{}'.format('model_intercept:', model.intercept))
        print('{}{}'.format('model_coeff:', model.coefficients))
        feature_map=dict(zip(inputCols, model.coefficients))
        print("feature_map",feature_map)
        #inference
        predictions = model.transform(pred_data)

        #as TempView to hive table
        predictions.select("shop_number", "item_number", "dt","prediction").createOrReplaceTempView('linear_predict_out')
        insert_sql="""insert overwrite table scm.linear_regression_prediction partition (dt)
        select
        store_code,
        goods_code,
        prediction,
        dt
        from 
        linear_predict_out"""
        spark.sql(insert_sql)
        spark.stop()


def read_importance_feature():
    """
    :return: list of importance of feature
    """

    importance_feature = spark.sql("""select feature from app.selection_result_v1 where cum_sum<0.95 and update_date 
    in (select max(update_date) as update_date from temp.selection_result_v1)""").select("feature").collect()
    importance_list = [row.feature for row in importance_feature]
    print('..use'+str(len(importance_list))+'numbers of feature...')
    return importance_list


def main():
    data=spark.sql("""select * from temp.dataset_feature'""")
    importance_feature=read_importance_feature()
    reg, inter, elastic = 0.5,False,1.0
    linear_predict(data, importance_feature, reg, inter, elastic).prediction()


if __name__ == '__main__':
    main()



