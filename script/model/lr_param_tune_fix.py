# encoding: utf-8
# Author    : huangjisheng
# Datetime  : 2021/4/17 14:55
# Product   : PyCharm
# File      : lr_param_tune.py

"""
this example ues LinearRegression,you can also tuning other model such as gbdt,lightgbm
two ways for tune param for spark.ml
1.CrossValidator
2.TrainValidationSplit
"""


import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import Normalizer
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegresion
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *
from pyspark.sql.types import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

spark = SparkSession. \
    Builder(). \
    config("spark.sql.crossJoin.enabled", "true"). \
    config("spark.sql.execution.arrow.enabled", "false"). \
    enableHiveSupport(). \
    getOrCreate()


def read_importance_feature():
    """
    :return: list of importance of feature
    """
    importance_feature = spark.sql("""select feature from app.selection_result where cum_sum<0.99 order by update_date desc,update_time desc limit 1""").select("feature").collect()
    importance_list = [row.feature for row in importance_feature]
    print('..use' + str(len(importance_list)) + 'numbers of feature...')
    return importance_list


class best_param(object):
    def __init__(self, is_cv=True,sample_ratio=0.1):
        self.today = (datetime.datetime.today()).strftime('%Y-%m-%d')
        self.update_time = str(datetime.datetime.now().strftime('%b-%m-%y %H:%M:%S')).split(' ')[-1]
        self.sample_ratio = sample_ratio
        self.is_cv=is_cv

    def run_tune(self, importance_feature, data):
        sample_ratio = self.sample_ratio
        df = data.sample(fraction=sample_ratio, seed=1688)
        df = df.na.fill(0)

        ssembler = VectorAssembler(inputCols=importance_feature, outputCol="non_norm_features")
        output = ssembler.transform(df)

        normalizer = Normalizer(inputCol='non_norm_features', outputCol="features")
        l1NormData = normalizer.transform(output, {normalizer.p: float(2)})
        features_label = l1NormData.select("features", "label")
        # split the dataset random
        train_data, test_data = features_label.randomSplit([0.8, 0.2])

        # train the model
        lr_params = ({'regParam': 0.00}, {'fitIntercept': True}, {'elasticNetParam': 0.5})
        lr = LinearRegression(maxIter=100, regParam=lr_params[0]['regParam'], \
                              fitIntercept=lr_params[1]['fitIntercept'], \
                              elasticNetParam=lr_params[2]['elasticNetParam'])

        model = lr.fit(train_data)
        pred = model.evaluate(test_data)

        # model of evaluate
        eval = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="mae")
        bef_mae = eval.evaluate(pred.predictions, {eval.metricName: "mae"})
        r2 = eval.evaluate(pred.predictions, {eval.metricName: "r2"})
        print('r2....' + str(r2))
        print('mae....' + str(bef_mae))

        lrParamGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.005, 0.01, 0.1, 0.5]) \
            .addGrid(lr.fitIntercept, [False, True]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.5, 1.0]) \
            .build()

        #build model Estimator such as LinearRegression
        #set RegressionEvaluator
        #fit model use k-fold,calculate avg of evaluate
        #save the best param

        if self.is_cv:
            train_valid = CrossValidator(estimator=lr, estimatorParamMaps=lrParamGrid, evaluator=eval,numFolds=5)
            tune_model = train_valid.fit(train_data)
            best_parameters = [(
                [{key.name: paramValue} for key, paramValue in zip(params.keys(), params.values())], metric) \
                for params, metric in zip(
                    tune_model.getEstimatorParamMaps(),
                    tune_model.avgMetrics)]

        else:
            train_valid = TrainValidationSplit(estimator=lr,estimatorParamMaps=lrParamGrid,evaluator=eval)
            tune_model = train_valid.fit(train_data)
            best_parameters = [(
                [{key.name: paramValue} for key, paramValue in zip(params.keys(), params.values())], metric) \
                for params, metric in zip(
                    tune_model.getEstimatorParamMaps(),
                    tune_model.validationMetrics)]

        lr_best_params = sorted(best_parameters, key=lambda el: el[1], reverse=True)[0][0]
        regParam_ky=[i for i in lr_best_params if i.get('regParam')][0]
        elasticNetParam_ky=[i for i in lr_best_params if i.get('elasticNetParam')][0]
        if [i for i in lr_best_params if i.get('fitIntercept')] is True:
            fitIntercept_ky = [i for i in check_d if i.get('fitIntercept')][0]
        else:
            fitIntercept_ky={'fitIntercept':False}

        pd_best_params = pd.DataFrame({
            'regParam': [regParam_ky['regParam']],
            'elasticNetParam': [elasticNetParam_ky['elasticNetParam']],
            'fitIntercept': [fitIntercept_ky['fitIntercept']]})

        pd_best_params['update_date'] = self.today
        pd_best_params['update_time'] = self.update_time
        pd_best_params['model_type'] = 'linear'

        # use the best param to predict
        lr = LinearRegression(maxIter=100, regParam=float(regParam_ky['regParam']),
                              elasticNetParam=float(elasticNetParam_ky['elasticNetParam']),
                              fitIntercept=bool(fitIntercept_ky['fitIntercept']))

        model = lr.fit(train_data)
        print('....intercept....' + str(model.intercept))
        print('....coefficients....' + str(model.coefficients))
        pred = model.evaluate(test_data)
        # evaluation model
        eval = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="mae")

        r2_tune = eval.evaluate(pred.predictions, {eval.metricName: "r2"})
        tune_mae = eval.evaluate(pred.predictions, {eval.metricName: "mae"})
        pd_best_params['bef_mae'] = str(bef_mae)
        pd_best_params['tune_mae'] = str(tune_mae)
        pd_best_params['tune_r2'] = str(r2_tune)
        pd_best_params = pd_best_params[['regParam', 'fitIntercept', 'elasticNetParam', 'model_type', 'bef_mae',
         'tune_mae', 'tune_r2','update_date', 'update_time']]
        if pd_best_params.shape[0]<1:
            raise ValueError("tune the best param is wrong at {},{}".format(self.today, self.update_time))
        pd_best_params=spark.createDataFrame(pd_best_params)
        try:
            pd_best_params.write.mode("append").format('hive').saveAsTable('app.regression_model_best_param')
        except:
            pd_best_params.createOrReplaceTempView('pd_best_params')
            spark.sql("""drop table if exists app.regression_model_best_param""")
            spark.sql("""create table app.regression_model_best_param as select * from pd_best_params""")

def main():
    importance_list = read_importance_feature()
    df = spark.sql("""select * from app.dataset_input_df_v2 where dt>='2020-08-24'""")
    best_param(is_cv=False).run_tune(importance_list, df)

if __name__ == '__main__':
    main()


