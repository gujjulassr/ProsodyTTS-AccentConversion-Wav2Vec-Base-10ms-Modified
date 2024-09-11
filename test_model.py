import tensorflow as tf
from tqdm import tqdm
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Multiply, Dense, Conv1D, GRU, Bidirectional, BatchNormalization, concatenate, MaxPool1D, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from swan import MHAttn
import os,sys
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import datetime

import warnings
import logging



# Create a logs directory with the current timestamp
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Suppress TensorFlow logs completely
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'







# Load dataset
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'my_dataset',
    split=['train', 'val', 'test'],
    shuffle_files=True,
    as_supervised=False,
    with_info=True,
)

# # Apply padding to the batches
# ds_train = ds_train.padded_batch(
#     40,
#     padded_shapes={
#         'features': [None, 768],
#         'mels': [None,80],
#         'mel_length': [],
#         'len_mask': [None,1],
#     }
# ).prefetch(tf.data.experimental.AUTOTUNE)

# ds_val = ds_val.padded_batch(
#     40,
#     padded_shapes={
#         'features': [None, 768],
#         'mels': [None,80],
#         'mel_length': [],
#         'len_mask': [None,1],
#     }
# ).prefetch(tf.data.experimental.AUTOTUNE)


# Apply padding to the batches
ds_train = ds_train.padded_batch(
    40,
    padded_shapes={
        'features': [None, 768],
        'mels': [None, 80],
        'mel_length': [],
        'len_mask': [None, 1],
    }
).repeat().prefetch(tf.data.experimental.AUTOTUNE)  # Adding .repeat() here

ds_val = ds_val.padded_batch(
    40,
    padded_shapes={
        'features': [None, 768],
        'mels': [None, 80],
        'mel_length': [],
        'len_mask': [None, 1],
    }
).repeat().prefetch(tf.data.experimental.AUTOTUNE)  # Adding .repeat() here


# Get the number of samples
num_train_samples = ds_info.splits['train'].num_examples
num_val_samples = ds_info.splits['val'].num_examples

# Calculate steps per epoch
batch_size = 40
steps_per_epoch = num_train_samples // batch_size
validation_steps = num_val_samples // batch_size

print(f"Number of training samples: {num_train_samples}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Number of validation samples: {num_val_samples}")
print(f"Validation steps: {validation_steps}")













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



x = acoustic_decoder(features_input)



x = new_acoustic_decoder(x, n_blocks=3, n_heads=4, head_size=64, context=10, inter_dim=128, out_dim=128)

est_mel = Dense(80, activation='relu')(x)
mel_gate = Dense(80, activation='sigmoid')(x)
est_mel = Multiply()([est_mel, mel_gate])

est_mel = Multiply(name='mel')([est_mel, len_mask])


model = Model(inputs=[features_input, len_mask], outputs=est_mel)



lr = 0.001
optimizer = Adam(learning_rate=lr)


model_check = ModelCheckpoint('weights_mel/weights-{epoch:04d}.keras', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=100)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, min_lr=1e-8, verbose=True)

# Initialize metrics
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

# Compile the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Checkpoint management
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory='weights_mel', max_to_keep=1)

# Best validation loss
best_val_loss = float('inf')

# Training and validation step functions
@tf.function
def train_step(features, len_mask, mels):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model([features, len_mask], training=True)
        loss = loss_function(mels, predictions)
    # Backward pass
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update metrics
    train_loss_metric(loss)
    return loss

@tf.function
def val_step(features, len_mask, mels):
    # Forward pass
    predictions = model([features, len_mask], training=False)
    loss = loss_function(mels, predictions)

    # Update metrics
    val_loss_metric(loss)
    return loss



epochs = 300
steps_per_epoch = 262
validation_steps = 32

best_val_loss = float('inf')
early_stop_patience = 100
early_stop_wait = 0

# Create a TensorBoard writer
train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # Reset metrics
    train_loss_metric.reset_state()
    val_loss_metric.reset_state()

    # Training
    for step, batch in enumerate(tqdm(ds_train), start=1):
        features = batch['features']
        len_mask = batch['len_mask']
        mels = batch['mels']
        loss = train_step(features, len_mask, mels)
        print(f"Batch {step}/{steps_per_epoch}: Training Loss: {loss.numpy()}")

        # Log the loss for each step
        with train_summary_writer.as_default():
            tf.summary.scalar('step_train_loss', loss, step=epoch * steps_per_epoch + step)

        # Break the loop after the specified number of steps
        if step >= steps_per_epoch:
            break

    # Log the average training loss for the epoch
    with train_summary_writer.as_default():
        tf.summary.scalar('epoch_train_loss', train_loss_metric.result(), step=epoch)

    # Validation
    for step, batch in enumerate(tqdm(ds_val), start=1):
        features = batch['features']
        len_mask = batch['len_mask']
        mels = batch['mels']
        loss = val_step(features, len_mask, mels)
        print(f"Batch {step}/{validation_steps}: Validation Loss: {loss.numpy()}")

        # Log the loss for each step
        with val_summary_writer.as_default():
            tf.summary.scalar('step_val_loss', loss, step=epoch * validation_steps + step)

        # Break the loop after the specified number of steps
        if step >= validation_steps:
            break

    # Log the average validation loss for the epoch
    with val_summary_writer.as_default():
        tf.summary.scalar('epoch_val_loss', val_loss_metric.result(), step=epoch)

    # Log epoch results
    print(f"Epoch {epoch+1}: Training Loss: {train_loss_metric.result().numpy()}, Validation Loss: {val_loss_metric.result().numpy()}")

    # Save the model if the validation loss has improved
    current_val_loss = val_loss_metric.result()
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        checkpoint_manager.save()
        print(f"Validation loss improved to {best_val_loss:.4f}. Saving model...")
        early_stop_wait = 0  # Reset patience counter
    else:
        early_stop_wait += 1

    # Early stopping logic
    if early_stop_wait >= early_stop_patience:
        print("Early stopping triggered")
        break

    # Reduce learning rate on plateau
    if reduce_lr.monitor_op(current_val_loss, reduce_lr.best):
        reduce_lr._reset()
    else:
        reduce_lr.wait += 1
        if reduce_lr.wait >= reduce_lr.patience:
            new_lr = max(reduce_lr.min_lr, reduce_lr.factor * float(tf.keras.backend.get_value(optimizer.learning_rate)))
            tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
            reduce_lr.wait = 0

# Save the final model after training
model.save('acoustic_decoder_model_final.h5')