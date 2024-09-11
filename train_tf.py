import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Multiply, Dense, Conv1D, GRU, Bidirectional, BatchNormalization, concatenate, MaxPool1D, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import numpy as np
from swan import MHAttn
import os,sys
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from datetime import datetime




# Load dataset
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'my_dataset',
    split=['train', 'val', 'test'],
    shuffle_files=True,
    as_supervised=False,
    with_info=True,
)

# Apply padding to the batches
ds_train = ds_train.padded_batch(
    40,
    padded_shapes={
        'features': [None, 768],
        'mels': [None,80],
        'mel_length': [],
        'len_mask': [None,1],
    }
).prefetch(tf.data.experimental.AUTOTUNE)

ds_val = ds_val.padded_batch(
    40,
    padded_shapes={
        'features': [None, 768],
        'mels': [None,80],
        'mel_length': [],
        'len_mask': [None,1],
    }
).prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.padded_batch(
    40,
    padded_shapes={
        'features': [None, 768],
        'mels': [None,80],
        'mel_length': [],
        'len_mask': [None,1],
    }
).prefetch(tf.data.experimental.AUTOTUNE)


def convert_to_tuple(features):
    # Assuming features is a dictionary with keys: 'features', 'len_mask', 'mels'
   
    return (features['features'], features['len_mask']), features['mels']


ds_train = ds_train.map(convert_to_tuple)
ds_val = ds_val.map(convert_to_tuple)
ds_test = ds_test.map(convert_to_tuple)



# for batch in ds_train:
#     print((batch[0]))
#     sys.exit()










def new_acoustic_decoder(inputs, n_blocks=1, n_heads=4, head_size=64, context=10, inter_dim=128, out_dim=128):
    x = Dense(inter_dim, activation='relu')(inputs)
    for i in range(n_blocks):
        cx =  MHAttn(n_heads, head_size, context)(x)
        x = BatchNormalization()(x+cx)
        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)                                    
    
    x = Dense(out_dim, activation='relu')(x)
    return x

def acoustic_decoder(inputs):
   
    conv_bank=[]
    for i in range(1,11):
        x = Conv1D(filters=128, kernel_size=i+1, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        conv_bank.append(x)

    x = concatenate(conv_bank, axis=-1)
    x = MaxPool1D(pool_size=2, strides=1, padding='same')(x) 
  
    
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(64, return_sequences=True))(x)
    return x





features_input = Input(shape=(None, 768), name='features')
len_mask = Input(shape=(None, 1), name='len_mask')

# Pass through the acoustic decoder
x = acoustic_decoder(features_input)

# Pass through the new acoustic decoder
x = new_acoustic_decoder(x, n_blocks=3, n_heads=4, head_size=64, context=10, inter_dim=128, out_dim=128)

# Predict mel spectrogram
est_mel = Dense(80, activation='relu')(x)

# Predict gate
mel_gate = Dense(80, activation='sigmoid')(x)


# Multiply mel and gate
est_mel = Multiply()([est_mel, mel_gate])



# Apply len_mask to est_mel
est_mel = Multiply(name='mel')([est_mel, len_mask])


# Create the model
model = Model(inputs=[features_input, len_mask], outputs=est_mel)
model.summary()


# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=MeanAbsoluteError(),
    metrics=[MeanAbsoluteError()]
)

# Define the path where you want to save the best model
best_model_path = 'best_model.keras'

# Define the ModelCheckpoint callback to save the best model
model_checkpoint = ModelCheckpoint(
    best_model_path,          # Path to save the model
    monitor='val_loss',       # Metric to monitor
    save_best_only=True,      # Only save the model if val_loss improves
    save_weights_only=False,  # Save the entire model (not just the weights)
    verbose=1                 # Verbosity mode
)



# Directory where the TensorBoard logs will be saved
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1, 
    write_graph=True, 
    write_images=True,
    update_freq='batch'  # Log metrics at every batch
)


# Define other callbacks as needed
early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, min_lr=1e-8, verbose=1,min_delta=0)

# Train the model with the callbacks
history = model.fit(
    ds_train,                 # Training data
    validation_data=ds_val,   # Validation data
    epochs=300,               # Number of epochs
    callbacks=[model_checkpoint, early_stop, reduce_lr,tensorboard_callback],  # List of callbacks
    verbose=1                 # Verbosity mode
)