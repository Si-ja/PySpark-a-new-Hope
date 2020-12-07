#%%
"""
In this instance the model that has been created previously will be re-used.
It is used in a seperate environment, just to emulate the moving of the model
happening between environments. For it, with pip, same packages as previously 
in 03. instance where installed + pyspark. But whatever
else got installed and comes with python, should be found
in the requirements.txt

Let's move into it. The output of each instance will be printed out.
"""
import pyspark
import sklearn
import pandas as pd
import numpy as np

print(f"Numpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Sklearn version: {sklearn.__version__}")
print(f"PySpark version: {pyspark.__version__}")

"""
Should yield:

In this instance the model that has been created previously will be re-used....
Numpy version: 1.19.1
Pandas version: 1.1.3
Sklearn version: 0.23.2
PySpark version: 3.0.1
"""

"""
Next things that need to be done;
1. SparkContext should be initiated.
2. The trained model should be loaded.
3. The data for processing should be generated or loaded.
4. Actually make predictions.
"""

# Step 1
from pyspark.sql import SparkSession
from pyspark import SparkContext

# Hopefully another context doesn't run
sc = SparkContext("local", "Prediction Model")
print(sc.master) 

"""
This should yiled:
local
"""

# Step 2.
# Because we were using a RandomForestClassifier we need to load that type of a Model
from pyspark.ml.classification import RandomForestClassificationModel
model = RandomForestClassificationModel.load("RF_model")

# Step 3.
from sklearn import datasets

# load the data we used
bc_dataset = datasets.load_breast_cancer()
data = bc_dataset.data
target = bc_dataset.target
features = bc_dataset.feature_names

# Arange the data
df = pd.DataFrame(data = data, columns = features)
df["targets"] = target

"""
Let's prettend we have a random new input of data
Only do this for testing purposes
Select a random value in the middle of the dataframe
"""
some_data = df.iloc[[int(len(df)/2)]]

"""
We don't want our targets, but we want to know
what we know our model will be predicting.
"""
truth = some_data["targets"]
some_data.drop("targets", axis=1)

# Arrange the data to be in the old format of the spark dataframe
from pyspark.sql import SparkSession, SQLContext
sqlCtx = SQLContext(sc)
sdf = sqlCtx.createDataFrame(some_data)

# Transform the data into a features format
from pyspark.ml.feature import VectorAssembler
required_features = list(some_data.columns)
assembler = VectorAssembler(inputCols=required_features, outputCol='features')
transformed_data = assembler.transform(sdf)

# Check if our data selected only has 1 observation
print(f"Training Dataset count: {transformed_data.count()}")

# Finally make predictions
predictions = model.transform(transformed_data)
prediction_form = predictions.toPandas()["prediction"]

print("\nResults of using Spark:")
print(f"Truth:\t\t {truth.to_numpy()}")
print(f"Predictions:\t {prediction_form.to_numpy()}")

"""
The predictions I got and hopefully you as well:
Results of using Spark:
Truth:		 [1]
Predictions: [1.]
"""


"""
When finished using the spark context - stop it
This will release the context without breaking anything
For the future insances
"""
sc.stop()

# %%
