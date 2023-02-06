#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

# Setting Configurations - JAVA_HOME and SPARK_HOME Variable

os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk1.8.0_251"
os.environ["SPARK_HOME"] = r"C:\Users\ASUS\Desktop\fiverracc2222\spark-3.2.1-bin-hadoop3.2"


# ## Creating a Spark Context

# In[2]:


# Import required modules

from pyspark.sql import functions
import pandas as pd


# In[3]:


import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# # 1.	Import the MovieLens dataset.

# In[4]:


from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# Load and parse the data
data = spark.sparkContext.textFile("u.data")
ratings = data.map(lambda l: l.split('\t')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

llist = ratings.collect()

for line in llist:
    print (line)


# # 2.	Build a recommendation model using Alternating Least Squares.

# In[5]:


# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)


# In[6]:


# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)


# # 3.	Report the original performance (Mean Squared Error)

# In[7]:


MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))


# # 4.	Try to improve the performance of the original model using 10-fold cross validation and solve the cold-start problem.
# 
# #### Used  CrossValidator(estimator = als, evaluator = evaluator, numFolds = 10) to add  10-fold cross validation
# #### Used coldStartStrategy="drop" to solve the cold-start problem by setting cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

# # 5.	Optimize the model based on step 4 and report the improvement of performance.

# In[12]:


from pyspark.ml.tuning import CrossValidator   # To add k-fold cross validation
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder

param_grid = ParamGridBuilder().addGrid(als.rank, [5, 40, 80, 120]).addGrid(als.maxIter, [5, 100, 250, 500]).addGrid(als.regParam, [.05, .1, 1.5]).build()

# Here used 5 max iterations and set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=5, regParam=0.01, userCol="user", itemCol="product", ratingCol="rating",coldStartStrategy="drop")

lines = spark.read.text("u.data").rdd
parts = lines.map(lambda row: row.value.split("\t"))

parts_list = parts.collect()
ratingsRDD = parts.map(lambda p: Rating(int(p[0]), int(p[1]), float(p[2])))
ratings = spark.createDataFrame(ratingsRDD)

(training, test) = ratings.randomSplit([0.8, 0.2])

# Tell Spark how to evaluate model performance
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")

# Build cross validation step using CrossValidator
cv = CrossValidator(estimator = als, estimatorParamMaps = param_grid, evaluator = evaluator, numFolds = 10)

model = cv.fit(training)      # Run the cv on the training data

best_model = model.bestModel  # Extract best combination of values from cross validation


# In[14]:


# Evaluate the model by computing the RMSE on the test data
predictions = best_model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("RMSE after optimization = " + str(rmse))


# ## 6.	Output top 10 movies for all the users with the following format:

# In[16]:


# Generate top 10 movie recommendations for each user
userRecs = best_model.recommendForAllUsers(10)

def get_used_id_and_corresponding_recommendations(x):
    user_id = x.user
    recommendations = x.recommendations
    return (user_id, recommendations)

rdd2 = userRecs.rdd.map(lambda x: get_used_id_and_corresponding_recommendations(x))


# In[17]:


## Process the spark dataframe for top 10 movie recommendations for each user and create the string to write to output file

used_id_recommendations = rdd2.collect()

final_str = ""
for line in used_id_recommendations:
    row_str = ""
    user_id = line[0]
    row_str = str(user_id) + "\t"
    recommendations = line[1]
    recommendation_movie_list = []
    for recommendation in recommendations:
        recommendation_movie_list.append(str(recommendation.product))
    row_str += ",". join(recommendation_movie_list)
    print (row_str)
    final_str = final_str + row_str + "\n"
    
final_str = final_str.strip()  


# In[18]:


## Writing output string to output.txt file

output_file = open("output.txt", "w")
output_file.write(final_str)
output_file.close()


# In[ ]:




