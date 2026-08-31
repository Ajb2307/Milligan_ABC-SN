import sys
import importlib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_fscore_support, classification_report
from sklearn.utils import class_weight
from keras.utils import to_categorical 
import tensorflow as tf
from tqdm.keras import TqdmCallback

sys.path.insert(0, '/lustre/lrspec/users/4301/hxe-for-tda/hxetda/')
import hxetda

sys.path.insert(0, "/lustre/lrspec/users/4301/ABC-SN/code")
import data_degrading as dg
import abcsn_training
import abcsn_config
import data_preparation as dp
import preprocessing as pp

sys.path.insert(0, "/lustre/lrspec/users/4301/Milligan_ABC-SN/")
from plot_data import *
from process_data import *
import heir_loss

import tensorflow as tf

# Get the build information dictionary
build_info = tf.sysconfig.get_build_info()

file = "/lustre/lrspec/users/4301/Milligan_ABC-SN/data/full_preprocessed_dataframe.parquet"
master_df = pd.read_parquet(file)

data = dp.extract_dataframe(master_df)

wvl = data[1]  # Wavelength array
flux0_columns = data[2]  # Columns that index the fluxes in the dataframe
metadata = data[5]  # Sub-dataframe containing only the metadata
all_features = data[6]  # Only the flux values in a numpy array
fluxes0 = data[6]  # Only the flux values in a numpy array

labels = metadata.sn_type
vertices =  np.asarray(np.unique(labels),dtype=str)

# Set up heirarchy
# Make a graph

G=nx.DiGraph()

G.add_edge('SN-like', 'SN')

G.add_edge('SN-like', 'Non-SN') 

# non-SN Transients
# G.add_edge('Non-SN', 'TDE')
# G.add_edge('Non-SN', 'CaRT')

# SN Ia
G.add_edge('SN', 'Ia')
G.add_edge('Ia', 'Iap')
G.add_edge('Ia', 'Ia-norm')


# CC SN
G.add_edge('SN', 'CC')
# G.add_edge('CC', 'SL')


## SN Ib/c
# G.add_edge('CC', 'SN Ib/c')
# G.add_edge('SN Ib/c', 'Ib')
# G.add_edge('SN Ib/c', 'Ic')
# G.add_edge('SN Ib/c', 'IIb')  # do under stripped envelope

## SN II
# G.add_edge('CC', 'II')
# # G.add_edge('CC', 'IIn') # not sure why this is separate from SN II 
# G.add_edge('II', 'IIn')
# G.add_edge('II', 'IIp') # this is essentially normal "plateau"

importlib.reload(hxetda)
pos = hxetda.hierarchy_pos(G, 'SN-like')

fig = plt.figure(1, figsize=(20, 10))

nx.draw_networkx(G, pos=pos, node_color='white', with_labels=False, node_size=2000, arrows=False)
text = nx.draw_networkx_labels(G, pos)
for _, t in text.items():
    t.set_rotation(45) 


# Load data + preprocessing (hxe-tda)
### for my current code all only need pathlengths, masklist, y_dict, and weights(?)

conversions = {'Ia':'Ia-norm',
                'II':'IIp',
               'CaRT': 'Non-SN',
               'TDE': 'Non-SN',
               'II': 'CC',
               'IIb': 'CC', 
               'IIn': 'CC',
               'Ib': 'CC',
               'Ic': 'CC', 
               'SL': 'CC'}


# Let's import preprocess our variable star data
scaler = StandardScaler()

def make_dataset(load=False):
    data = dp.extract_dataframe(master_df)
    
    metadata = data[5]  # Sub-dataframe containing only the metadata
    flux0_columns = data[2]  # Columns that index the fluxes in the dataframe
    all_labels = metadata.sn_type.values
    vertices =  np.asarray(np.unique(all_labels),dtype=str)
    all_flux = data[6] 

    for key in conversions:
        gind = np.where(all_labels == key)
        if len(gind[0])==0:
            print(f"no labels: {key}")
        else:
            all_labels[gind] = conversions[key]
    
    vertices = np.asarray(np.unique(all_labels),dtype=str)
    print(np.unique(all_labels, return_counts=True))
    vertices = np.append(vertices, ['SN', "Ia"])
    vertices = np.insert(vertices, 0, 'SN-like')

    return(all_labels, vertices)

labels, vertices = make_dataset()
scaler.fit(all_features)
all_features = scaler.transform(all_features)

np.unique(labels)
# classification leaves 
leaves = np.asarray(['CC', 'Ia-norm', 'Iap', 'Non-SN'], dtype=str)
# Some pre-processing for the graph
paths, pathlengths, mask_list, y_dict = hxetda.calc_path_and_mask(G, vertices, 'SN-like')

# Set up training and model
index = master_df.index.values
y_mhe = np.array([y_dict[x] for x in labels]) # "multi hot encoding"
X = fluxes0.copy()
X = X.reshape(X.shape[0],1,X.shape[1])

# Getting index of the train test split
train_index, test_index = train_test_split(index, test_size = 0.3, random_state = 7, stratify = labels)

# splitting all the inputs
X_train = X[train_index]
X_test = X[test_index]

y_train = y_mhe[train_index]
y_test = y_mhe[test_index]

labels_test = labels[test_index]

# Class weights
# Some pre-processing for the hxetda class weights
class_weight_hxetda_dict = hxetda.calc_class_weights(labels, vertices, paths)
class_weight_hxetda_dict

weights = np.array([class_weight_hxetda_dict[x] for x in labels])
weights_train = weights[train_index]
weights_test = weights[test_index]

# loss function 
mask_list = np.array(mask_list)
pathlengths = np.array(pathlengths)
loss_fn = heir_loss.WeightedHierchicalLoss_hxetda(mask_list, pathlengths, alpha=0.5)

early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        # min_delta=1e-2, # stated during meeting (note higher than willows)
        min_delta=1e-3, 
        patience= 25, # stated in meeting + same as willows
        verbose=2,
        mode="min",
        restore_best_weights=True 
        )

# model_filepath = "/lustre/lrspec/users/4301/Milligan_ABC-SN/models/model_checkpoint{epoch}.keras" # this one creates many files
model_filepath = "/lustre/lrspec/users/4301/Milligan_ABC-SN/models/model_checkpoint_weighted_loss.keras"
model_checkpoints = keras.callbacks.ModelCheckpoint(
    model_filepath,
    monitor="val_loss",
    verbose=0,
    save_freq = 10,
    mode="min",
    save_best_only=True
    )

epoch_bound = 100
batch_size = 64 # used below
step_bound = len(train_index) * epoch_bound // batch_size

learning_rate_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[step_bound],  # Epoch when the change happens
    values=[1e-5, 2e-6]  # Learning rates: [before boundary, after boundary]
    )

# Loading model 
abcsn = keras.models.load_model("/lustre/lrspec/users/4301/ABC-SN/abcsn/ABCSN.keras", compile=False)

# Fine-tune from this layer onwards
fine_tune_at = len(abcsn.layers)-8

# Freeze all the layers before the `fine_tune_at` layer
for layer in abcsn.layers[:fine_tune_at]:
    layer.trainable = False

new_model = keras.Model(
    inputs=abcsn.input, 
    outputs=abcsn.layers[-2].output)

from tensorflow.keras import layers

num_labels = 7
new_output = layers.Dense(num_labels, activation="softmax", name="heirarchical_dense")(new_model.output)
heir_abcsn = keras.Model(inputs=new_model.input, outputs=new_output)

acc = keras.metrics.CategoricalAccuracy(name="ca")
f1 = keras.metrics.F1Score(average="macro", name="f1")
optimizer = keras.optimizers.Nadam(learning_rate=learning_rate_schedule)

# Compile your model
heir_abcsn.compile(
    loss=loss_fn,
    optimizer=optimizer,
    metrics=[acc, f1]
)
def main():
    # weights are implemented incorrectly
    history = heir_abcsn.fit(X_train, y_train, 
                        sample_weight=weights_train,
                        validation_data= (X_test, y_test, weights_test), 
                        epochs = 800,    # choosen arbitrarily 
                        batch_size = 64, # same as willow used
                        verbose= 1,
                        callbacks=[model_checkpoints,  TqdmCallback(verbose=0)], # removed early stop
                    )

    model_name = "heirarchical_truncated_model"
    heir_abcsn.save(f"/lustre/lrspec/users/4301/Milligan_ABC-SN/model_results/{model_name}.keras")
    (pd.DataFrame(history.history)).to_csv(f"/lustre/lrspec/users/4301/Milligan_ABC-SN/model_results/{model_name}.csv")

if __name__ == "__main__":
    main()
