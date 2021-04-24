# encoding: utf-8
# Author    : huangjisheng
# Datetime  : 2020/9/16 13:30
# Product   : PyCharm
# File      : glm_model_v1.py


import datetime
from pyspark.sql import SparkSession
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

spark = SparkSession. \
    Builder(). \
    config("spark.sql.crossJoin.enabled", "true"). \
    config("spark.sql.execution.arrow.enabled", "false"). \
    enableHiveSupport(). \
    getOrCreate()

class generalized_linear(object):
    def __init__(self, data, importance_feature):
        self.data = data
        self.importance_feature = importance_feature
        self.today = (datetime.datetime.today()).strftime('%Y-%m-%d')
        self.update_time = str(datetime.datetime.now().strftime('%b-%m-%y %H:%M:%S')).split(' ')[-1]

    def prediction(self):
        df = self.data
        inputCols = self.importance_feature
        update_time = self.update_time
        df = df.na.fill(0)
        feature_vector = VectorAssembler(inputCols=inputCols, outputCol="features")
        output = feature_vector.transform(df)
        features_lable = output.select("shop_number", "item_number", "dt", "features", "label")
        train_set = features_lable.where(features_lable['dt'] <= '2020-08-28')
        train_data, val_data = train_set.randomSplit([0.8, 0.2])
        pred_data = features_lable.where(features_lable['dt'] > '2020-08-28').where(features_lable['dt'] < '2020-09-01')
        glr = GeneralizedLinearRegression(family="poisson", link="identity", regParam=1.0)
        model = glr.fit(train_data)
        print('{}{}'.format('model_intercept:', model.intercept))
        print('{}{}'.format('model_coeff:', model.coefficients))
        feature_map = dict(zip(inputCols[1], model.coefficients))
        print("feature_map", feature_map)
        pred = model.evaluate(val_data)
        # evaluation model
        eval = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="mae")
        val_mae = eval.evaluate(pred.predictions, {eval.metricName: "mae"})
        # model prediction
        predictions = model.transform(pred_data)
        predictions.select("shop_number", "item_number", "dt", "prediction").createOrReplaceTempView(
            'linear_predict_out')

        #执行预测数据插入
        predict_inset_sql="""
        insert overwrite table scm.generalized_linear_model_prediction partition(dt)
        select
        store_code,
        goods_code,
        prediction,
         {val_mae} as val_mae,
        {update_time} as update_time,
        dt
        from linear_predict_out"""
        predict_inset_sql = predict_inset_sql.format(val_mae=val_mae,update_time=update_time)
        spark.sql(predict_inset_sql)
        spark.stop()


def read_importance_feature():
    """
    :return: list of importance of feature
    """
    importance_feature = spark.sql("""select feature from app.selection_result_v1 where cum_sum<0.95 and update_date 
    in (select max(update_date) as update_date from temp.selection_result_v1)""").select("feature").collect()
    importance_list = [row.feature for row in importance_feature]
    print('..use' + str(len(importance_list)) + 'numbers of feature...')
    return importance_list


def main():
    data = spark.sql("""select * from temp.dataset_feature""")
    importance_feature = read_importance_feature()
    generalized_linear(data, importance_feature).prediction()


if __name__ == '__main__':
    main()



