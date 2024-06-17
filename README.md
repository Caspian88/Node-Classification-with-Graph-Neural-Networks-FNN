# Node-Classification-with-Graph-Neural-Networks-FNN

Introduction
Many datasets in various machine learning (ML) applications have structural relationships between their entities, which can be represented as graphs. Such application includes social and communication networks analysis, traffic prediction, and fraud detection. Graph representation Learning aims to build and train models for graph datasets to be used for a variety of ML tasks.

This example demonstrate a simple implementation of a Graph Neural Network (GNN) model. The model is used for a node prediction task on the Cora dataset to predict the subject of a paper given its words and citations network.

Note that, we implement a Graph Convolution Layer from scratch to provide better understanding of how they work. However, there is a number of specialized TensorFlow-based libraries that provide rich GNN APIs, such as Spectral, StellarGraph, and GraphNets.

Setup
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
Prepare the Dataset
The Cora dataset consists of 2,708 scientific papers classified into one of seven classes. The citation network consists of 5,429 links. Each paper has a binary word vector of size 1,433, indicating the presence of a corresponding word.

Download the dataset
The dataset has two tap-separated files: cora.cites and cora.content.

The cora.cites includes the citation records with two columns: cited_paper_id (target) and citing_paper_id (source).
The cora.content includes the paper content records with 1,435 columns: paper_id, subject, and 1,433 binary features.
Let's download the dataset.

zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
)
data_dir = os.path.join(os.path.dirname(zip_file), "cora")
Process and visualize the dataset
Then we load the citations data into a Pandas DataFrame.

citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)
print("Citations shape:", citations.shape)
Citations shape: (5429, 2)
Now we display a sample of the citations DataFrame. The target column includes the paper ids cited by the paper ids in the source column.

citations.sample(frac=1).head()
target	source
2581	28227	6169
1500	7297	7276
1194	6184	1105718
4221	139738	1108834
3707	79809	1153275
Now let's load the papers data into a Pandas DataFrame.

column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"), sep="\t", header=None, names=column_names,
)
print("Papers shape:", papers.shape)
Papers shape: (2708, 1435)
Now we display a sample of the papers DataFrame. The DataFrame includes the paper_id and the subject columns, as well as 1,433 binary column representing whether a term exists in the paper or not.

print(papers.sample(5).T)
                    1                133                    2425  \
paper_id         1061127            34355                1108389   
term_0                 0                0                      0   
term_1                 0                0                      0   
term_2                 0                0                      0   
term_3                 0                0                      0   
...                  ...              ...                    ...   
term_1429              0                0                      0   
term_1430              0                0                      0   
term_1431              0                0                      0   
term_1432              0                0                      0   
subject    Rule_Learning  Neural_Networks  Probabilistic_Methods   
                         2103             1346  
paper_id              1153942            80491  
term_0                      0                0  
term_1                      0                0  
term_2                      1                0  
term_3                      0                0  
...                       ...              ...  
term_1429                   0                0  
term_1430                   0                0  
term_1431                   0                0  
term_1432                   0                0  
subject    Genetic_Algorithms  Neural_Networks  
[1435 rows x 5 columns]
Let's display the count of the papers in each subject.

print(papers.subject.value_counts())
Neural_Networks           818
Probabilistic_Methods     426
Genetic_Algorithms        418
Theory                    351
Case_Based                298
Reinforcement_Learning    217
Rule_Learning             180
Name: subject, dtype: int64
We convert the paper ids and the subjects into zero-based indices.

class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])
Now let's visualize the citation graph. Each node in the graph represents a paper, and the color of the node corresponds to its subject. Note that we only show a sample of the papers in the dataset.

plt.figure(figsize=(10, 10))
colors = papers["subject"].tolist()
cora_graph = nx.from_pandas_edgelist(citations.sample(n=1500))
subjects = list(papers[papers["paper_id"].isin(list(cora_graph.nodes))]["subject"])
nx.draw_spring(cora_graph, node_size=15, node_color=subjects)
png

Split the dataset into stratified train and test sets
train_data, test_data = [], []

for _, group_data in papers.groupby("subject"):
    # Select around 50% of the dataset for training.
    random_selection = np.random.rand(len(group_data.index)) <= 0.5
    train_data.append(group_data[random_selection])
    test_data.append(group_data[~random_selection])

train_data = pd.concat(train_data).sample(frac=1)
test_data = pd.concat(test_data).sample(frac=1)

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
Train data shape: (1360, 1435)
Test data shape: (1348, 1435)
Implement Train and Evaluate Experiment
hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256
This function compiles and trains an input model using the given training data.

def run_experiment(model, x_train, y_train):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )

    return history
This function displays the loss and accuracy curves of the model during training.

def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()
Implement Feedforward Network (FFN) Module
We will use this module in the baseline and the GNN models.

def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)
Build a Baseline Neural Network Model
Prepare the data for the baseline model
feature_names = list(set(papers.columns) - {"paper_id", "subject"})
num_features = len(feature_names)
num_classes = len(class_idx)

# Create train and test features as a numpy array.
x_train = train_data[feature_names].to_numpy()
x_test = test_data[feature_names].to_numpy()
# Create train and test targets as a numpy array.
y_train = train_data["subject"]
y_test = test_data["subject"]
Implement a baseline classifier
We add five FFN blocks with skip connections, so that we generatee a baseline model with roughly the same number of parameters as the GNN models to be built later.

def create_baseline_model(hidden_units, num_classes, dropout_rate=0.2):
    inputs = layers.Input(shape=(num_features,), name="input_features")
    x = create_ffn(hidden_units, dropout_rate, name=f"ffn_block1")(inputs)
    for block_idx in range(4):
        # Create an FFN block.
        x1 = create_ffn(hidden_units, dropout_rate, name=f"ffn_block{block_idx + 2}")(x)
        # Add skip connection.
        x = layers.Add(name=f"skip_connection{block_idx + 2}")([x, x1])
    # Compute logits.
    logits = layers.Dense(num_classes, name="logits")(x)
    # Create the model.
    return keras.Model(inputs=inputs, outputs=logits, name="baseline")


baseline_model = create_baseline_model(hidden_units, num_classes, dropout_rate)
baseline_model.summary()
Model: "baseline"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_features (InputLayer)     [(None, 1433)]       0                                            
__________________________________________________________________________________________________
ffn_block1 (Sequential)         (None, 32)           52804       input_features[0][0]             
__________________________________________________________________________________________________
ffn_block2 (Sequential)         (None, 32)           2368        ffn_block1[0][0]                 
__________________________________________________________________________________________________
skip_connection2 (Add)          (None, 32)           0           ffn_block1[0][0]                 
                                                                 ffn_block2[0][0]                 
__________________________________________________________________________________________________
ffn_block3 (Sequential)         (None, 32)           2368        skip_connection2[0][0]           
__________________________________________________________________________________________________
skip_connection3 (Add)          (None, 32)           0           skip_connection2[0][0]           
                                                                 ffn_block3[0][0]                 
__________________________________________________________________________________________________
ffn_block4 (Sequential)         (None, 32)           2368        skip_connection3[0][0]           
__________________________________________________________________________________________________
skip_connection4 (Add)          (None, 32)           0           skip_connection3[0][0]           
                                                                 ffn_block4[0][0]                 
__________________________________________________________________________________________________
ffn_block5 (Sequential)         (None, 32)           2368        skip_connection4[0][0]           
__________________________________________________________________________________________________
skip_connection5 (Add)          (None, 32)           0           skip_connection4[0][0]           
                                                                 ffn_block5[0][0]                 
__________________________________________________________________________________________________
logits (Dense)                  (None, 7)            231         skip_connection5[0][0]           
==================================================================================================
Total params: 62,507
Trainable params: 59,065
Non-trainable params: 3,442
__________________________________________________________________________________________________
Train the baseline classifier
history = run_experiment(baseline_model, x_train, y_train)
Epoch 1/300
5/5 [==============================] - 3s 203ms/step - loss: 4.1695 - acc: 0.1660 - val_loss: 1.9008 - val_acc: 0.3186
Epoch 2/300
5/5 [==============================] - 0s 15ms/step - loss: 2.9269 - acc: 0.2630 - val_loss: 1.8906 - val_acc: 0.3235
Epoch 3/300
5/5 [==============================] - 0s 15ms/step - loss: 2.5669 - acc: 0.2424 - val_loss: 1.8713 - val_acc: 0.3186
Epoch 4/300
5/5 [==============================] - 0s 15ms/step - loss: 2.1377 - acc: 0.3147 - val_loss: 1.8687 - val_acc: 0.3529
Epoch 5/300
5/5 [==============================] - 0s 15ms/step - loss: 2.0256 - acc: 0.3297 - val_loss: 1.8285 - val_acc: 0.3235
Epoch 6/300
5/5 [==============================] - 0s 15ms/step - loss: 1.8148 - acc: 0.3495 - val_loss: 1.8000 - val_acc: 0.3235
Epoch 7/300
5/5 [==============================] - 0s 15ms/step - loss: 1.7216 - acc: 0.3883 - val_loss: 1.7771 - val_acc: 0.3333
Epoch 8/300
5/5 [==============================] - 0s 15ms/step - loss: 1.6941 - acc: 0.3910 - val_loss: 1.7528 - val_acc: 0.3284
Epoch 9/300
5/5 [==============================] - 0s 15ms/step - loss: 1.5690 - acc: 0.4358 - val_loss: 1.7128 - val_acc: 0.3333
Epoch 10/300
5/5 [==============================] - 0s 15ms/step - loss: 1.5139 - acc: 0.4367 - val_loss: 1.6650 - val_acc: 0.3676
Epoch 11/300
5/5 [==============================] - 0s 15ms/step - loss: 1.4370 - acc: 0.4930 - val_loss: 1.6145 - val_acc: 0.3775
Epoch 12/300
5/5 [==============================] - 0s 15ms/step - loss: 1.3696 - acc: 0.5109 - val_loss: 1.5787 - val_acc: 0.3873
Epoch 13/300
5/5 [==============================] - 0s 15ms/step - loss: 1.3979 - acc: 0.5341 - val_loss: 1.5564 - val_acc: 0.3922
Epoch 14/300
5/5 [==============================] - 0s 15ms/step - loss: 1.2681 - acc: 0.5599 - val_loss: 1.5547 - val_acc: 0.3922
Epoch 15/300
5/5 [==============================] - 0s 16ms/step - loss: 1.1970 - acc: 0.5807 - val_loss: 1.5735 - val_acc: 0.3873
Epoch 16/300
5/5 [==============================] - 0s 15ms/step - loss: 1.1555 - acc: 0.6032 - val_loss: 1.5131 - val_acc: 0.4216
Epoch 17/300
5/5 [==============================] - 0s 15ms/step - loss: 1.1234 - acc: 0.6130 - val_loss: 1.4385 - val_acc: 0.4608
Epoch 18/300
5/5 [==============================] - 0s 14ms/step - loss: 1.0507 - acc: 0.6306 - val_loss: 1.3929 - val_acc: 0.4804
Epoch 19/300
5/5 [==============================] - 0s 15ms/step - loss: 1.0341 - acc: 0.6393 - val_loss: 1.3628 - val_acc: 0.4902
Epoch 20/300
5/5 [==============================] - 0s 35ms/step - loss: 0.9457 - acc: 0.6693 - val_loss: 1.3383 - val_acc: 0.4902
Epoch 21/300
5/5 [==============================] - 0s 17ms/step - loss: 0.9054 - acc: 0.6756 - val_loss: 1.3365 - val_acc: 0.4951
Epoch 22/300
5/5 [==============================] - 0s 15ms/step - loss: 0.8952 - acc: 0.6854 - val_loss: 1.3228 - val_acc: 0.5049
Epoch 23/300
5/5 [==============================] - 0s 15ms/step - loss: 0.8413 - acc: 0.7217 - val_loss: 1.2924 - val_acc: 0.5294
Epoch 24/300
5/5 [==============================] - 0s 15ms/step - loss: 0.8543 - acc: 0.6998 - val_loss: 1.2379 - val_acc: 0.5490
Epoch 25/300
5/5 [==============================] - 0s 16ms/step - loss: 0.7632 - acc: 0.7376 - val_loss: 1.1516 - val_acc: 0.5833
Epoch 26/300
5/5 [==============================] - 0s 15ms/step - loss: 0.7189 - acc: 0.7496 - val_loss: 1.1296 - val_acc: 0.5931
Epoch 27/300
5/5 [==============================] - 0s 15ms/step - loss: 0.7433 - acc: 0.7482 - val_loss: 1.0937 - val_acc: 0.6127
Epoch 28/300
5/5 [==============================] - 0s 15ms/step - loss: 0.7310 - acc: 0.7440 - val_loss: 1.0950 - val_acc: 0.5980
Epoch 29/300
5/5 [==============================] - 0s 16ms/step - loss: 0.7059 - acc: 0.7654 - val_loss: 1.1343 - val_acc: 0.5882
Epoch 30/300
5/5 [==============================] - 0s 21ms/step - loss: 0.6831 - acc: 0.7645 - val_loss: 1.1938 - val_acc: 0.5686
Epoch 31/300
5/5 [==============================] - 0s 23ms/step - loss: 0.6741 - acc: 0.7788 - val_loss: 1.1281 - val_acc: 0.5931
Epoch 32/300
5/5 [==============================] - 0s 16ms/step - loss: 0.6344 - acc: 0.7753 - val_loss: 1.0870 - val_acc: 0.6029
Epoch 33/300
5/5 [==============================] - 0s 16ms/step - loss: 0.6052 - acc: 0.7876 - val_loss: 1.0947 - val_acc: 0.6127
Epoch 34/300
5/5 [==============================] - 0s 15ms/step - loss: 0.6313 - acc: 0.7908 - val_loss: 1.1186 - val_acc: 0.5882
Epoch 35/300
5/5 [==============================] - 0s 16ms/step - loss: 0.6163 - acc: 0.7955 - val_loss: 1.0899 - val_acc: 0.6176
Epoch 36/300
5/5 [==============================] - 0s 16ms/step - loss: 0.5388 - acc: 0.8203 - val_loss: 1.1222 - val_acc: 0.5882
Epoch 37/300
5/5 [==============================] - 0s 16ms/step - loss: 0.5487 - acc: 0.8080 - val_loss: 1.0205 - val_acc: 0.6127
Epoch 38/300
5/5 [==============================] - 0s 16ms/step - loss: 0.5885 - acc: 0.7903 - val_loss: 0.9268 - val_acc: 0.6569
Epoch 39/300
5/5 [==============================] - 0s 15ms/step - loss: 0.5541 - acc: 0.8025 - val_loss: 0.9367 - val_acc: 0.6471
Epoch 40/300
5/5 [==============================] - 0s 36ms/step - loss: 0.5594 - acc: 0.7935 - val_loss: 0.9688 - val_acc: 0.6275
Epoch 41/300
5/5 [==============================] - 0s 17ms/step - loss: 0.5255 - acc: 0.8169 - val_loss: 1.0076 - val_acc: 0.6324
Epoch 42/300
5/5 [==============================] - 0s 16ms/step - loss: 0.5284 - acc: 0.8180 - val_loss: 1.0106 - val_acc: 0.6373
Epoch 43/300
5/5 [==============================] - 0s 15ms/step - loss: 0.5141 - acc: 0.8188 - val_loss: 0.8842 - val_acc: 0.6912
Epoch 44/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4767 - acc: 0.8342 - val_loss: 0.8249 - val_acc: 0.7108
Epoch 45/300
5/5 [==============================] - 0s 15ms/step - loss: 0.5915 - acc: 0.8055 - val_loss: 0.8567 - val_acc: 0.6912
Epoch 46/300
5/5 [==============================] - 0s 15ms/step - loss: 0.5026 - acc: 0.8357 - val_loss: 0.9287 - val_acc: 0.6618
Epoch 47/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4859 - acc: 0.8304 - val_loss: 0.9044 - val_acc: 0.6667
Epoch 48/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4860 - acc: 0.8440 - val_loss: 0.8672 - val_acc: 0.6912
Epoch 49/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4723 - acc: 0.8358 - val_loss: 0.8717 - val_acc: 0.6863
Epoch 50/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4831 - acc: 0.8457 - val_loss: 0.8674 - val_acc: 0.6912
Epoch 51/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4873 - acc: 0.8353 - val_loss: 0.8587 - val_acc: 0.7010
Epoch 52/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4537 - acc: 0.8472 - val_loss: 0.8544 - val_acc: 0.7059
Epoch 53/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4684 - acc: 0.8425 - val_loss: 0.8423 - val_acc: 0.7206
Epoch 54/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4436 - acc: 0.8523 - val_loss: 0.8607 - val_acc: 0.6961
Epoch 55/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4589 - acc: 0.8335 - val_loss: 0.8462 - val_acc: 0.7059
Epoch 56/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4757 - acc: 0.8360 - val_loss: 0.8415 - val_acc: 0.7010
Epoch 57/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4270 - acc: 0.8593 - val_loss: 0.8094 - val_acc: 0.7255
Epoch 58/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4530 - acc: 0.8307 - val_loss: 0.8357 - val_acc: 0.7108
Epoch 59/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4370 - acc: 0.8453 - val_loss: 0.8804 - val_acc: 0.7108
Epoch 60/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4379 - acc: 0.8465 - val_loss: 0.8791 - val_acc: 0.7108
Epoch 61/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4254 - acc: 0.8615 - val_loss: 0.8355 - val_acc: 0.7059
Epoch 62/300
5/5 [==============================] - 0s 15ms/step - loss: 0.3929 - acc: 0.8696 - val_loss: 0.8355 - val_acc: 0.7304
Epoch 63/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4039 - acc: 0.8516 - val_loss: 0.8576 - val_acc: 0.7353
Epoch 64/300
5/5 [==============================] - 0s 35ms/step - loss: 0.4220 - acc: 0.8596 - val_loss: 0.8848 - val_acc: 0.7059
Epoch 65/300
5/5 [==============================] - 0s 17ms/step - loss: 0.4091 - acc: 0.8521 - val_loss: 0.8560 - val_acc: 0.7108
Epoch 66/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4658 - acc: 0.8470 - val_loss: 0.8518 - val_acc: 0.7206
Epoch 67/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4269 - acc: 0.8437 - val_loss: 0.7878 - val_acc: 0.7255
Epoch 68/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4368 - acc: 0.8438 - val_loss: 0.7859 - val_acc: 0.7255
Epoch 69/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4113 - acc: 0.8452 - val_loss: 0.8056 - val_acc: 0.7402
Epoch 70/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4304 - acc: 0.8469 - val_loss: 0.8093 - val_acc: 0.7451
Epoch 71/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4159 - acc: 0.8585 - val_loss: 0.8090 - val_acc: 0.7451
Epoch 72/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4218 - acc: 0.8610 - val_loss: 0.8028 - val_acc: 0.7402
Epoch 73/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3632 - acc: 0.8714 - val_loss: 0.8153 - val_acc: 0.7304
Epoch 74/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3745 - acc: 0.8722 - val_loss: 0.8299 - val_acc: 0.7402
Epoch 75/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3997 - acc: 0.8680 - val_loss: 0.8445 - val_acc: 0.7255
Epoch 76/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4143 - acc: 0.8620 - val_loss: 0.8344 - val_acc: 0.7206
Epoch 77/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4006 - acc: 0.8616 - val_loss: 0.8358 - val_acc: 0.7255
Epoch 78/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4266 - acc: 0.8532 - val_loss: 0.8266 - val_acc: 0.7206
Epoch 79/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4337 - acc: 0.8523 - val_loss: 0.8181 - val_acc: 0.7206
Epoch 80/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3857 - acc: 0.8624 - val_loss: 0.8143 - val_acc: 0.7206
Epoch 81/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4146 - acc: 0.8567 - val_loss: 0.8192 - val_acc: 0.7108
Epoch 82/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3638 - acc: 0.8794 - val_loss: 0.8248 - val_acc: 0.7206
Epoch 83/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4126 - acc: 0.8678 - val_loss: 0.8565 - val_acc: 0.7255
Epoch 84/300
5/5 [==============================] - 0s 36ms/step - loss: 0.3941 - acc: 0.8530 - val_loss: 0.8624 - val_acc: 0.7206
Epoch 85/300
5/5 [==============================] - 0s 17ms/step - loss: 0.3843 - acc: 0.8786 - val_loss: 0.8389 - val_acc: 0.7255
Epoch 86/300
5/5 [==============================] - 0s 15ms/step - loss: 0.3651 - acc: 0.8747 - val_loss: 0.8314 - val_acc: 0.7206
Epoch 87/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3911 - acc: 0.8657 - val_loss: 0.8736 - val_acc: 0.7255
Epoch 88/300
5/5 [==============================] - 0s 15ms/step - loss: 0.3706 - acc: 0.8714 - val_loss: 0.9159 - val_acc: 0.7108
Epoch 89/300
5/5 [==============================] - 0s 15ms/step - loss: 0.4403 - acc: 0.8386 - val_loss: 0.9038 - val_acc: 0.7206
Epoch 90/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3865 - acc: 0.8668 - val_loss: 0.8733 - val_acc: 0.7206
Epoch 91/300
5/5 [==============================] - 0s 15ms/step - loss: 0.3757 - acc: 0.8643 - val_loss: 0.8704 - val_acc: 0.7157
Epoch 92/300
5/5 [==============================] - 0s 15ms/step - loss: 0.3828 - acc: 0.8669 - val_loss: 0.8786 - val_acc: 0.7157
Epoch 93/300
5/5 [==============================] - 0s 15ms/step - loss: 0.3651 - acc: 0.8787 - val_loss: 0.8977 - val_acc: 0.7206
Epoch 94/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3913 - acc: 0.8614 - val_loss: 0.9415 - val_acc: 0.7206
Epoch 95/300
5/5 [==============================] - 0s 15ms/step - loss: 0.3995 - acc: 0.8590 - val_loss: 0.9495 - val_acc: 0.7157
Epoch 96/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4228 - acc: 0.8508 - val_loss: 0.9490 - val_acc: 0.7059
Epoch 97/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3853 - acc: 0.8789 - val_loss: 0.9402 - val_acc: 0.7157
Epoch 98/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3711 - acc: 0.8812 - val_loss: 0.9283 - val_acc: 0.7206
Epoch 99/300
5/5 [==============================] - 0s 15ms/step - loss: 0.3949 - acc: 0.8578 - val_loss: 0.9591 - val_acc: 0.7108
Epoch 100/300
5/5 [==============================] - 0s 15ms/step - loss: 0.3563 - acc: 0.8780 - val_loss: 0.9744 - val_acc: 0.7206
Epoch 101/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3579 - acc: 0.8815 - val_loss: 0.9358 - val_acc: 0.7206
Epoch 102/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4069 - acc: 0.8698 - val_loss: 0.9245 - val_acc: 0.7157
Epoch 103/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3161 - acc: 0.8955 - val_loss: 0.9401 - val_acc: 0.7157
Epoch 104/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3346 - acc: 0.8910 - val_loss: 0.9517 - val_acc: 0.7157
Epoch 105/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4204 - acc: 0.8538 - val_loss: 0.9366 - val_acc: 0.7157
Epoch 106/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3492 - acc: 0.8821 - val_loss: 0.9424 - val_acc: 0.7353
Epoch 107/300
5/5 [==============================] - 0s 16ms/step - loss: 0.4002 - acc: 0.8604 - val_loss: 0.9842 - val_acc: 0.7157
Epoch 108/300
5/5 [==============================] - 0s 35ms/step - loss: 0.3701 - acc: 0.8736 - val_loss: 0.9999 - val_acc: 0.7010
Epoch 109/300
5/5 [==============================] - 0s 17ms/step - loss: 0.3391 - acc: 0.8866 - val_loss: 0.9768 - val_acc: 0.6961
Epoch 110/300
5/5 [==============================] - 0s 15ms/step - loss: 0.3857 - acc: 0.8739 - val_loss: 0.9953 - val_acc: 0.7255
Epoch 111/300
5/5 [==============================] - 0s 16ms/step - loss: 0.3822 - acc: 0.8731 - val_loss: 0.9817 - val_acc: 0.7255
Epoch 112/300
5/5 [==============================] - 0s 23ms/step - loss: 0.3211 - acc: 0.8887 - val_loss: 0.9781 - val_acc: 0.7108
Epoch 113/300
5/5 [==============================] - 0s 20ms/step - loss: 0.3473 - acc: 0.8715 - val_loss: 0.9927 - val_acc: 0.6912
Epoch 114/300
5/5 [==============================] - 0s 20ms/step - loss: 0.4026 - acc: 0.8621 - val_loss: 1.0002 - val_acc: 0.6863
Epoch 115/300
5/5 [==============================] - 0s 20ms/step - loss: 0.3413 - acc: 0.8837 - val_loss: 1.0031 - val_acc: 0.6912
Epoch 116/300
5/5 [==============================] - 0s 20ms/step - loss: 0.3653 - acc: 0.8765 - val_loss: 1.0065 - val_acc: 0.7010
Epoch 117/300
5/5 [==============================] - 0s 21ms/step - loss: 0.3147 - acc: 0.8974 - val_loss: 1.0206 - val_acc: 0.7059
Epoch 118/300
5/5 [==============================] - 0s 21ms/step - loss: 0.3639 - acc: 0.8783 - val_loss: 1.0206 - val_acc: 0.7010
Epoch 119/300
5/5 [==============================] - 0s 19ms/step - loss: 0.3660 - acc: 0.8696 - val_loss: 1.0260 - val_acc: 0.6912
Epoch 120/300
5/5 [==============================] - 0s 18ms/step - loss: 0.3624 - acc: 0.8708 - val_loss: 1.0619 - val_acc: 0.6814
Let's plot the learning curves.

display_learning_curves(history)
png

Now we evaluate the baseline model on the test data split.

_, test_accuracy = baseline_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
Test accuracy: 73.52%
Examine the baseline model predictions
Let's create new data instances by randomly generating binary word vectors with respect to the word presence probabilities.

def generate_random_instances(num_instances):
    token_probability = x_train.mean(axis=0)
    instances = []
    for _ in range(num_instances):
        probabilities = np.random.uniform(size=len(token_probability))
        instance = (probabilities <= token_probability).astype(int)
        instances.append(instance)

    return np.array(instances)


def display_class_probabilities(probabilities):
    for instance_idx, probs in enumerate(probabilities):
        print(f"Instance {instance_idx + 1}:")
        for class_idx, prob in enumerate(probs):
            print(f"- {class_values[class_idx]}: {round(prob * 100, 2)}%")
Now we show the baseline model predictions given these randomly generated instances.

new_instances = generate_random_instances(num_classes)
logits = baseline_model.predict(new_instances)
probabilities = keras.activations.softmax(tf.convert_to_tensor(logits)).numpy()
display_class_probabilities(probabilities)
Instance 1:
- Case_Based: 13.02%
- Genetic_Algorithms: 6.89%
- Neural_Networks: 23.32%
- Probabilistic_Methods: 47.89%
- Reinforcement_Learning: 2.66%
- Rule_Learning: 1.18%
- Theory: 5.03%
Instance 2:
- Case_Based: 1.64%
- Genetic_Algorithms: 59.74%
- Neural_Networks: 27.13%
- Probabilistic_Methods: 9.02%
- Reinforcement_Learning: 1.05%
- Rule_Learning: 0.12%
- Theory: 1.31%
Instance 3:
- Case_Based: 1.35%
- Genetic_Algorithms: 77.41%
- Neural_Networks: 9.56%
- Probabilistic_Methods: 7.89%
- Reinforcement_Learning: 0.42%
- Rule_Learning: 0.46%
- Theory: 2.92%
Instance 4:
- Case_Based: 0.43%
- Genetic_Algorithms: 3.87%
- Neural_Networks: 92.88%
- Probabilistic_Methods: 0.97%
- Reinforcement_Learning: 0.56%
- Rule_Learning: 0.09%
- Theory: 1.2%
Instance 5:
- Case_Based: 0.11%
- Genetic_Algorithms: 0.17%
- Neural_Networks: 10.26%
- Probabilistic_Methods: 0.5%
- Reinforcement_Learning: 0.35%
- Rule_Learning: 0.63%
- Theory: 87.97%
Instance 6:
- Case_Based: 0.98%
- Genetic_Algorithms: 23.37%
- Neural_Networks: 70.76%
- Probabilistic_Methods: 1.12%
- Reinforcement_Learning: 2.23%
- Rule_Learning: 0.21%
- Theory: 1.33%
Instance 7:
- Case_Based: 0.64%
- Genetic_Algorithms: 2.42%
- Neural_Networks: 27.19%
- Probabilistic_Methods: 14.07%
- Reinforcement_Learning: 1.62%
- Rule_Learning: 9.35%
- Theory: 44.7%
Build a Graph Neural Network Model
Prepare the data for the graph model
Preparing and loading the graphs data into the model for training is the most challenging part in GNN models, which is addressed in different ways by the specialised libraries. In this example, we show a simple approach for preparing and using graph data that is suitable if your dataset consists of a single graph that fits entirely in memory.

The graph data is represented by the graph_info tuple, which consists of the following three elements:

node_features: This is a [num_nodes, num_features] NumPy array that includes the node features. In this dataset, the nodes are the papers, and the node_features are the word-presence binary vectors of each paper.
edges: This is [num_edges, num_edges] NumPy array representing a sparse adjacency matrix of the links between the nodes. In this example, the links are the citations between the papers.
edge_weights (optional): This is a [num_edges] NumPy array that includes the edge weights, which quantify the relationships between nodes in the graph. In this example, there are no weights for the paper citations.
# Create an edges array (sparse adjacency matrix) of shape [2, num_edges].
edges = citations[["source", "target"]].to_numpy().T
# Create an edge weights array of ones.
edge_weights = tf.ones(shape=edges.shape[1])
# Create a node features array of shape [num_nodes, num_features].
node_features = tf.cast(
    papers.sort_values("paper_id")[feature_names].to_numpy(), dtype=tf.dtypes.float32
)
# Create graph info tuple with node_features, edges, and edge_weights.
graph_info = (node_features, edges, edge_weights)

print("Edges shape:", edges.shape)
print("Nodes shape:", node_features.shape)
Edges shape: (2, 5429)
Nodes shape: (2708, 1433)
Implement a graph convolution layer
We implement a graph convolution module as a Keras Layer. Our GraphConvLayer performs the following steps:

Prepare: The input node representations are processed using a FFN to produce a message. You can simplify the processing by only applying linear transformation to the representations.
Aggregate: The messages of the neighbours of each node are aggregated with respect to the edge_weights using a permutation invariant pooling operation, such as sum, mean, and max, to prepare a single aggregated message for each node. See, for example, tf.math.unsorted_segment_sum APIs used to aggregate neighbour messages.
Update: The node_repesentations and aggregated_messages—both of shape [num_nodes, representation_dim]— are combined and processed to produce the new state of the node representations (node embeddings). If combination_type is gru, the node_repesentations and aggregated_messages are stacked to create a sequence, then processed by a GRU layer. Otherwise, the node_repesentations and aggregated_messages are added or concatenated, then processed using a FFN.
The technique implemented use ideas from Graph Convolutional Networks, GraphSage, Graph Isomorphism Network, Simple Graph Networks, and Gated Graph Sequence Neural Networks. Two other key techniques that are not covered are Graph Attention Networks and Message Passing Neural Networks.

def create_gru(hidden_units, dropout_rate):
    inputs = keras.layers.Input(shape=(2, hidden_units[0]))
    x = inputs
    for units in hidden_units:
      x = layers.GRU(
          units=units,
          activation="tanh",
          recurrent_activation="sigmoid",
          return_sequences=True,
          dropout=dropout_rate,
          return_state=False,
          recurrent_dropout=dropout_rate,
      )(x)
    return keras.Model(inputs=inputs, outputs=x)


class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        if self.combination_type == "gru":
            self.update_fn = create_gru(hidden_units, dropout_rate)
    else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim]
        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_repesentations
        )
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)
Implement a graph neural network node classifier
The GNN classification model follows the Design Space for Graph Neural Networks approach, as follows:

Apply preprocessing using FFN to the node features to generate initial node representations.
Apply one or more graph convolutional layer, with skip connections, to the node representation to produce node embeddings.
Apply post-processing using FFN to the node embeddings to generate the final node embeddings.
Feed the node embeddings in a Softmax layer to predict the node class.
Each graph convolutional layer added captures information from a further level of neighbours. However, adding many graph convolutional layer can cause oversmoothing, where the model produces similar embeddings for all the nodes.

Note that the graph_info passed to the constructor of the Keras model, and used as a property of the Keras model object, rather than input data for training or prediction. The model will accept a batch of node_indices, which are used to lookup the node features and neighbours from the graph_info.

class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        # Create a process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )
        # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2",
        )
        # Create a postprocess layer.
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        # Create a compute logits layer.
        self.compute_logits = layers.Dense(units=num_classes, name="logits")

    def call(self, input_node_indices):
        # Preprocess the node_features to produce node representations.
        x = self.preprocess(self.node_features)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x1 + x
        # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x2 + x
        # Postprocess node embedding.
        x = self.postprocess(x)
        # Fetch node embeddings for the input node_indices.
        node_embeddings = tf.gather(x, input_node_indices)
        # Compute logits
        return self.compute_logits(node_embeddings)
Let's test instantiating and calling the GNN model. Notice that if you provide N node indices, the output will be a tensor of shape [N, num_classes], regardless of the size of the graph.
