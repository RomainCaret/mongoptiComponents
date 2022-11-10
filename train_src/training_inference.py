import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
from dtreeviz.trees import *
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

# Handling the parsing of the arguments/parameters
parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--max_epocs", type=int, help="Max # of epocs for the training")
parser.add_argument("--learning_rate", type=float, help="Learning rate")
parser.add_argument("--learning_rate_schedule", type=str, help="Learning rate schedule")
parser.add_argument("--model_output", type=str, help="Path of output model")

args = parser.parse_args()

print("hello training world...")

lines = [
    f"Training data path: {args.training_data}",
    f"Max epocs: {args.max_epocs}",
    f"Learning rate: {args.learning_rate}",
    f"Learning rate: {args.learning_rate_schedule}",
    f"Model output path: {args.model_output}",
]

for line in lines:
    print(line)

print("mounted_path files: ")
arr_files = os.listdir(args.training_data)
print(arr_files)

# arr_files contains the list of files in the mounted path



# PLOTTING THE ACTUALL VS PREDICTED TIME.
def actual_vs_predicted(index, actual, predicted, text="CompletionTime"):


    plt.scatter(index, actual,marker="D")
    plt.scatter(index,predicted, color="r",marker="*")
    plt.legend(["Actual", "Predicted"])
    plt.title("Actual versus Predicted values for {}".format(text))
    plt.xlabel("Index", fontsize=20)
    plt.ylabel("completionTime", fontsize=20)
    return plt.figure()

def saveModel(path, model):
    pickle.dump(model, open(path, 'wb'))

 # The datavolume column has some strings which have to be converted into proper henc, the function

def dataVolumeConversion(val):
    
    if val[-1] == "K":
        return int(val[:-1])*1000
    if val[-1] == "M":
        return int(val[:-1])*1000000
 
# XCa change /dbfs/FileStore/tables/TrainingDataSetCWOMongoDB18SEPRun2search5ShreeSave
data = pd.read_csv(os.path.join(args.training_data, arr_files[0]))

print(data)
data.datavolume =  data.datavolume.apply(dataVolumeConversion)

cluster_type = {"flights-m10":0, 'flights-m20':1, 'flights-m50':2, 'flights-m80':3,
       'flights-m200':4}
data.MongoClusterType = data.MongoClusterType.map(lambda x:cluster_type[x])

#converitng the querytype from string to numeric
queryDict = {val:i for i, val in enumerate(data.querytype.unique())}
data.querytype = data.querytype.map(lambda x:queryDict[x])

#spillting the dataset 
X_data, y_data = data[["MongoClusterType", "RAM", "CPUs", "datavolume",'querytype','shardingtype', 'indextype', 'Memory', 'Iops']], data["responsetime"]
X_train, X_test, y_train, y_test = train_test_split( X_data, y_data, test_size=0.1, random_state=42)
#XCA remove iops and add querytype


# First predicting the time taken to do the write operation
regressor = DecisionTreeRegressor(random_state = 0, max_depth=7, criterion="mae") 
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_predict))
# Do the train and save the trained model as a file into the output folder.
# Here only output a dummy data for demo.
curtime = datetime.now().strftime("%b-%d-%Y %H:%M:%S")
model = f"This is a dummy model with id: {str(uuid4())} generated at: {curtime}\n"
saveModel((Path(args.model_output) / "decisionTree20Sep.sav"),regressor)