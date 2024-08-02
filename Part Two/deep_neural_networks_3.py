import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'mnist', # Specifies the MNIST dataset.
    split=['train', 'test'], # Splits the data into training and testing sets.
    shuffle_files=True, # Shuffles the dataset files.
    as_supervised=True, # Returns the dataset in a (image, label) format.
    with_info=True, # Returns other metadata
)

def normalize_img(image, label):
  """Normalizes images: `uint8` (0-255) -> `float32` (0.0-1.0)."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE) # applies the normalize_img function to each image in the training set
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples) # shuffles the dataset
ds_train = ds_train.batch(128) # batch elements of the dataset after shuffling to get unique batches at each epoch.
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)