from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, MinMaxScaler, VectorAssembler
from pyspark.sql.functions import corr
from pyspark.sql.functions import when, col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

#verify pyspark
spark = SparkSession.builder.appName("PySparkTest").getOrCreate()
print("PySpark Initialized:", spark.version)

#load dataset
df = spark.read.csv("/Users/cynthia/PycharmProjects/bigDataAssignment/bankdataset.csv", header=True, inferSchema=True)
#print("Number of rows:", df.count())
#print("Schema:")
#df.printSchema()

#data processing
#verify columns in dataset
print("Columns in the dataset: ")
print(df.columns)

#missing value handling
df = df.na.fill({"Value": 0, "Transaction_count": 0})

#categorical data encoding
indexer = StringIndexer(inputCol="Domain", outputCol="domain_index")
df = indexer.fit(df).transform(df)

indexer_location = StringIndexer(inputCol="Location", outputCol="location_index")
df = indexer_location.fit(df).transform(df)

#normalize numerical data
numerical_columns = ["Value","Transaction_count"]
assembler = VectorAssembler(inputCols=numerical_columns, outputCol="features")
df = assembler.transform(df)

scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

#show processed data
df.show()

#eda performing
df.describe(["Value", "Transaction_count"]).show()

df.groupBy("Value").count().orderBy("Value").show()
df.groupBy("Transaction_count").count().orderBy("Transaction_count").show()

##correlation between variables
df.select(corr("Value", "Transaction_count")).show()

##data exporting
pandas_df = df.limit(1000).toPandas()
pandas_df.to_csv("sample.csv", index=False)

#load exported sample
df_sample = pd.read_csv('sample.csv')

##histogram for value
plt.figure(figsize=(8, 5))
sns.histplot(df_sample['Value'], kde=True, bins=30)
plt.title("Distribution of Value")
plt.xlabel("Value")
plt.ylabel("frequency")
plt.show()

##boxplot for transaction_count
plt.figure(figsize=(8,5))
sns.boxplot(x=df_sample['Transaction_count'])
plt.title("Boxplot of Transaction Count")
plt.show()

#2 algorithms
##logistic regression
mean_value = df.select("transaction_count").groupBy().mean().collect()[0][0]
df = df.withColumn("label", when(col("Transaction_count") > mean_value, 1).otherwise(0))
##spilt the data
train_df, test_df = df.randomSplit([0.7, 0.3], seed = 42)
##train logistic regression model
lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")
lr_model = lr.fit(train_df)
##evaluate model
predictions = lr_model.transform(test_df)
predictions.select("label", "prediction").show()

##k-means clustering
kmeans = KMeans(featuresCol="scaled_features", k=3)
kmeans_model = kmeans.fit(df)
print("Cluster Centers: ", kmeans_model.clusterCenters())

df = kmeans_model.transform(df)
df.select("scaled_features", "prediction").show()

#performance optimization
##enabling cache
df.cache()
##data repartition
df = df.repartition(10) #cluster size
##compare execution times
start = time.time()
df.describe().show()
end = time.time()
print("Execution:", end - start)