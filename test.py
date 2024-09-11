import tensorflow_datasets as tfds
import sys
import tensorflow as tf


(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'my_dataset',
    split=['train', 'val', 'test'],
    shuffle_files=True,
    as_supervised=False,
    with_info=True,
)


ds_train = ds_train.padded_batch(
    2,
    padded_shapes={
        'features': [None, 768],
        'mels': [None,80],
        'mel_length': [],
        'len_mask': [None,1],
    }
).prefetch(tf.data.experimental.AUTOTUNE)


for batch in ds_train:
    print(batch['len_mask'])
    sys.exit()