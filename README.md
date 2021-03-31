# Temporal Transformer

Temporal Transformer is a time-sensitive data transformation package for dynamic machine learning. Temporal Transformer generates 3 dimensional tensors for each entity's features for all of its respective observation periods.

# Step 1: Creating a Hopper object

First, we simply create a hopper object with the following line of code. When creating the Hopper object, we can set verbose to be true or false.
```python
h = Hopper.dbms(verbose=True)
```

# Step 2: Creating and adding table configurations

Next, we create table configurations. This will simply tell Hopper how to create tables. For each table configuration, we give a table name, column names, column types, indicate if the data has primary keys, and indicate if the data has temporal columns.
Important usage notes:
* The entity IDs must be the first column and will not be passed in as a column name or type, regardless of whether the dataset has primary keys or not.
* If there are time columns, they must be immediately after the entity id column.
* Datasets with time columns can have two columns to represent an observation period or have 1 time column to represent a timestamp or the exact time when an observation was recorded
* Indicate if the time data is in unix time (hasUnixTimes = ) and whether there's 1 column for time or 2 (hasTimestamps = )
* Available column types: bin (binary), real, ldc (low dimensional category), hdc (high dimensional category)

```python
# Creating a table configuration
_tc = Hopper.table_config(
    "AAPL", # Table name
    ["Open", "High", "Low", "Close", "Volume", "market_up"], # Column names
    ["real", "real", "real", "real", "real", "bin"], # Respective column types
    has_times=True,
    primary_keys=False
)

# Adding table configuration to Hopper object
h.create_fvm_with_csv(_tc, "aaple_stock_data.csv", hasUnixTimes=False, hasTimestamps=True, delimeter=',')
```

# Step 3: Hopper's dew_it function

The last step for Hopper is calling the dew_it function. This function windows, partitions, aggregates, and normalizes data. This is the last step of Hopper before handing the Hopper object off to Prepper. **The dt argument allows users to specify a time window they would like the data to be aggregated by.** The dt argument is flexible, allowing users to use unix time or a string for the value. For example, the following would all be valid values for dt. dt can be any unit of time: seconds, minutes, hours, days, weeks, months, years. Important to note that a month is considered exactly 30 days.
```python
dt="1 day"
dt="1 week"
dt="30 min"
dt="140 minutes"
dt="1 hr"
dt="2.5 hours"
```
The following line of code calls dew_it on the Hopper object with a given time window of 1 day.
```python
h.dew_it(fit_normalization_via_sql_qds=False,dt="1 days")
```

# Step 4: Creating a Prepper object

We now create a Prepper object from our Hopper object.
```python
tfp = Prepper.tf_prepper(h)
```

# Step 5: Prepper's fit function

Next, we use Prepper's fit function to initialize offsets and labels to predict. Offsets refer to how far in advance should predictions be made. We pass in a list of integers to accomplish this. The unit of time for the given offsets come from the value of dt used in dew_it. For example, if we pass in 1, 3, and 5 for our offsets, this will tell the model to make predictions 1, 3, and 5 days from each point in time.
```python
tfp.fit(offsets=[1,3,5], label_fns=["AAPL_avg_Close"], partition="train")
```
Here, we are telling Prepper to make predictions for 1, 3, and 5 days in the future for each point in time, and we are predicting the Close column from the table named AAPL. To get a list of all features, we can look at the features attribute of our Prepper object.
```python
tfp.features
```

# Step 6: Transforming the data

For the last step in the data transformation process, we use Prepper's transform_to_ds() function. After this, we can create our model and start making predictions.
```python
ds = tfp.transform_to_ds()
```

# Step 7: Building the model

The next step is to build a model with Prepper's build_model function. The following block of code uses a simple RNN. For the build_model function, we simply pass in a middle_layer_list and an activation type.
```python
d0 = tf.keras.layers.Dense(units=32, name="encode")
r0 = tf.keras.layers.LSTM(units=32,return_sequences=True, name="RNN_0")
r1 = tf.keras.layers.LSTM(units=16,return_sequences=True, name="RNN_1")
model = tfp.build_model(middle_layer_list=[d0, r0, r1], activation=None)
model.compile(loss="mean_absolute_error")
model.summary()
model.fit(ds["train"], epochs=10)
```

# Step 8: Prediction

The following block of code shows how to use make predictions and use Prepper's plot function. The plot function plots the actual values and the predicted values for each of the given offsets. We create a Prepper entity by passing in an entity ID and Prepper object. We can then use the entity's predict function, passing in our model, to make predictions. **It's important to note that the entity ID passed in when creating a Prepper entity object will tell Prepper to make predictions based off rows with the given entity ID.**

```python
e = Prepper.entity("AAPL", tfp)
e.predict(model)
e.plot()
```
To create your own graphs or dive deeper into the predictions, we can use the entity object's predictions attribute. This will return a dictionary where the key is the predicted label, and the values are a list of a lists. Each inner list will be the predicted values for respective offsets. For our example, index 0 of an inner list corresponds to the prediction for a 1 day offset. Index 2 would correspond to a prediction for a 5 day offset. For the outer list, each index refers to a point in time. For example, index 0 would correspond to the list of predictions made at time t=0. The following line of code allows us to see the predictions dictionary.
```python
e.predictions
```
