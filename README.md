# Temporal Transformer

Temporal Transformer is a time-sensitive data transformation package for dynamic machine learning. Temporal Transformer generates 3 dimensional tensors for each entity's features for all of its respective observation periods.

# Step 1: Hopper

The goal of Hopper is to prepare the data to be converted into tensors. Data is normalized over windows of time, which is specified by the user. This section will take you through the steps of Hopper.

### Step 1.1: Creating a Hopper object

First, we simply create a hopper object with the following line of code. When creating the Hopper object, we can set verbose to be true or false.
```python
h = Hopper.dbms(verbose=True)
```

### Step 1.2: Creating and adding table configurations

Next, we create table configurations. This will simply tell Hopper how to create tables. For each table configuration, we give a table name, column names, column types, indicate if the data has primary keys, and indicate if the data has temporal columns.
Important usage notes:
* The entity IDs must be the first column and will not be passed in as a column name or type, regardless of whether the dataset has primary keys or not.
* If there are time columns, they must be immediately after the entity id column.
* Datasets with time columns can have two columns to represent an observation period or have 1 time column to represent a timestamp or the exact time when an observation was recorded
* Indicate if the time data is in unix time (hasUnixTimes = ) and whether there's 1 column for time or 2 (hasTimestamps = )
* Available column types: bin (binary), real, ldc (low dimensional category), hdc (high dimensional category)

```python
# Creating a table configuration: characteristics table, no times with unique keys
_tc = Hopper.table_config("characteristics", 
                  ['Age', 'Weight', 'Gender', 'Height'], 
                  ['real', 'real', 'bin', 'real'],
                  has_times=False,
                  primary_key=True)
# Add table configuration to our Hopper object
h.create_fvm_with_csv(_tc, "characteristics.csv", delimiter=',')

# Creating a table configuration: has times but not unique keys
_tc = Hopper.table_config("vitals", 
                  ['RespRate', 'NISysABP', 'Weight', 'NIMAP', 'HR', 'Temp', 'MAP', 'DiasABP', 'NIDiasABP', 'SysABP'], 
                  ['real', 'real', 'real', 'real', 'real', 'real', 'real', 'real', 'real', 'real'],
                  has_times=True,
                  primary_key=False)
h.create_fvm_with_csv(_tc, "vitals.csv", hasUnixTimes=False, hasTimestamps=True, delimiter=',')

...

# Create and add all table configurations necessary

```

### Step 1.3: Hopper's dew_it function

The last step for Hopper is calling the dew_it function. This function windows, partitions, aggregates, and normalizes data. This is the last step of Hopper before handing the Hopper object off to Prepper. **The dt argument allows users to specify a time window they would like the data to be aggregated by.** The dt argument is flexible, allowing users to use unix time or a string for the value. For example, the following would all be valid values for dt. dt can be any unit of time: seconds, minutes, hours, days, weeks, months, years. The dt argument can also be an integer/float, which will either be interpreted as seconds or simply a number, depending on whether your data has actual dates/times or relative times. Important to note that a month is considered exactly 30 days.
```python
dt = "1 day"
dt = "1 week"
dt = "30 min"
dt = "140 minutes"
dt = "1 hr"
dt = "2.5 hours"
dt = 100
```
The following line of code calls dew_it on the Hopper object with a given time window of 100. We are using an integer here since the data we are working with has relative times, with most times for an entity ranging from 0 to 5000. View the google collab to examine the datasets.
```python
h.dew_it(fit_normalization_via_sql_qds=False,dt=100)
```

# Step 2: Prepper

Prepper is the second and last part of Temporal Transformer. Prepper takes the finished product from Hopper and converts it to tensors. The tensors have dimensions of entity id, feature name, and time. Predictions are then made for each offset at every time. More in depth explanation and steps of how to use Prepper are below.

### Step 2.1: Creating a Prepper object

We now create a Prepper object from our Hopper object.
```python
tfp = Prepper.tf_prepper(h)
```

### Step 2.2: Prepper's fit function

Next, we use Prepper's fit function to initialize offsets and labels to predict. Offsets refer to how far in advance should predictions be made. We pass in a list of integers to accomplish this. The unit of time for the given offsets come from the value of dt used in dew_it. For example, if we pass in 1, 2, and 3 for our offsets, this will tell the model to make predictions 100, 200, and 300 units of time from each point in time. This is because we passed in a dt of 100 back in step 1.3.
```python
tfp.fit(offsets=[1,2,3], label_fns=["deaths/avg_death"], partition="train")
```
Here, we are telling Prepper to make predictions for 100, 200, and 300 units of time in the future for each point in time, and we are predicting the death column from the table named deaths. To get a list of all features, we can look at the features attribute of our Prepper object. Looking at the features is helpful in figuring out the exact name of the feature you are passing into fit to predict.
```python
tfp.features
```

### Step 2.3: Transforming the data

For the last step in the data transformation process, we use Prepper's transform_to_ds() function. After this, we can create our model and start making predictions.
```python
ds = tfp.transform_to_ds()
```

### Step 2.4: Building the model

The next step is to build a model with Prepper's build_model function. The following block of code uses a simple RNN. For the build_model function, we simply pass in a middle_layer_list and an activation type.
```python
d0 = tf.keras.layers.Dense(units=32, name="encode")
r0 = tf.keras.layers.LSTM(units=32,return_sequences=True, name="RNN_0")
r1 = tf.keras.layers.LSTM(units=16,return_sequences=True, name="RNN_1")
model = tfp.build_model(middle_layer_list=[d0, r0, r1], activation=None)
model.compile(loss="binary_cross_entropy")
model.summary()
model.fit(ds["train"], epochs=10)
```

### Step 2.5: Prediction

The following block of code shows how to use make predictions and use Prepper's plot function. The plot function plots the actual values and the predicted values for each of the given offsets. We create a Prepper entity by passing in an entity ID and Prepper object. We can then use the entity's predict function, passing in our model, to make predictions. **It's important to note that the entity ID passed in when creating a Prepper entity object will tell Prepper to make predictions based off rows with the given entity ID.**

```python
# Predict for a single entity
e = Prepper.entity(132588, tfp)
e.predict(model)
e.plot()

# Predict for multiple entities
entities = [132588, 133166, 133588, 141068, 141510]
for entity in entities:
    e = Prepper.entity(entity, tfp)
    e.predict(model)
    e.plot()
```
To create your own graphs or dive deeper into the predictions, we can use the entity object's predictions attribute. This will return a dictionary where the key is the predicted label, and the values are a list of a lists. Each inner list will be the predicted values for respective offsets. For our example, index 0 of an inner list corresponds to the prediction for a 100 unit of time offset. Index 2 would correspond to a prediction for a 300 unit of time offset. For the outer list, each index refers to a point in time. For example, index 0 would correspond to the list of predictions made at time t=0. The following line of code allows us to see the predictions dictionary.
```python
e.predictions
```
