import tensorflow as tf
from tensorflow.keras import layers, models
import json
import pathlib
import os
from dotenv import load_dotenv
load_dotenv()

dataset_path = pathlib.Path(os.getenv('dataset_path'))
tf_dataset_output_dir = dataset_path / "spoken_digit_tf_datasets"
config_path = pathlib.Path(os.getenv('config_file_path'))
model_output_dir = pathlib.Path(os.getenv('model_path'))

try:
    with open(str(config_path), 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found.")
    exit()

INPUT_SHAPE = tuple(config['input_shape'])
NUM_CLASSES = config['num_classes']
EMBEDDING_DIM = config['conformer_blocks']['embedding_dim']
NUM_BLOCKS = config['conformer_blocks']['num_blocks']
FF_EXPANSION_FACTOR = config['feed_forward']['expansion_factor']
ATTN_NUM_HEADS = config['self_attention']['num_heads']
ATTN_KEY_DIM = config['self_attention']['key_dim']
CONV_KERNEL_SIZE = config['convolutional']['kernel_size']
DROPOUT_RATE = config['conformer_blocks']['dropout_rate']
LEARNING_RATE = config['training_params']['learning_rate']
EPOCHS = config['training_params']['epochs']

def build_conformer_model(input_shape, num_classes, embedding_dim, num_blocks, ff_expansion_factor, 
                          attn_num_heads, attn_key_dim, conv_kernel_size, dropout_rate):
    
    inputs = layers.Input(shape=input_shape, name="input_features")
    x = layers.Dense(embedding_dim)(inputs)
    x = layers.Dropout(dropout_rate)(x)

    for _ in range(num_blocks):
        ff_input = x
        x = layers.LayerNormalization()(x)
        x = layers.Dense(embedding_dim * ff_expansion_factor, activation="swish")(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(embedding_dim)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = x + ff_input

        mha_input = x
        x = layers.LayerNormalization()(x)
        x = layers.MultiHeadAttention(num_heads=attn_num_heads, key_dim=attn_key_dim, dropout=dropout_rate)(x, x, x)
        x = layers.Dropout(dropout_rate)(x)
        x = x + mha_input

        conv_input = x
        x = layers.LayerNormalization()(x)
        x = layers.Conv1D(filters=embedding_dim * 2, kernel_size=1, activation="relu")(x)
        x = layers.DepthwiseConv1D(kernel_size=conv_kernel_size, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=embedding_dim, kernel_size=1)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = x + conv_input

        ff2_input = x
        x = layers.LayerNormalization()(x)
        x = layers.Dense(embedding_dim * ff_expansion_factor, activation="swish")(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(embedding_dim)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = x + ff2_input
        
        x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train(FORCE_TRAIN = False):

    model_save_path = model_output_dir / f"{config['model_name']}.keras"
    if not FORCE_TRAIN and os.path.exists(model_save_path):
        print(f"Model already trained. Skipping.")
        return True, None
    
    try:
        train_ds_path = str(tf_dataset_output_dir / "train")
        val_ds_path = str(tf_dataset_output_dir / "validation")
        
        element_spec = (
            tf.TensorSpec(shape=(None, *INPUT_SHAPE), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
        
        train_dataset = tf.data.Dataset.load(train_ds_path, element_spec=element_spec)
        validation_dataset = tf.data.Dataset.load(val_ds_path, element_spec=element_spec)

    except Exception as e:
        print(f"Error loading datasets: {e}")
        return False, e
    
    try:
        conformer_model = build_conformer_model(
            INPUT_SHAPE, NUM_CLASSES, EMBEDDING_DIM, NUM_BLOCKS, FF_EXPANSION_FACTOR,
            ATTN_NUM_HEADS, ATTN_KEY_DIM, CONV_KERNEL_SIZE, DROPOUT_RATE
        )
        conformer_model.summary()
        tf.keras.utils.plot_model(conformer_model, to_file=str(model_output_dir / 'model_architecture.png'), show_shapes=True)

        history = conformer_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=EPOCHS
        )
        
        conformer_model.save(str(model_save_path))
        print("\nTraining finished.")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print("Model saved")

    except Exception as e:
        print(f"Training Failes due to {e}")
        return False, e

    try:
        metrics = {
            "final_training_metrics": {
                "loss": history.history['loss'][-1],
                "accuracy": history.history['accuracy'][-1]
            },
            "final_validation_metrics": {
                "val_loss": history.history['val_loss'][-1],
                "val_accuracy": history.history['val_accuracy'][-1]
            },
            "history": {
                "loss": history.history['loss'],
                "accuracy": history.history['accuracy'],
                "val_loss": history.history['val_loss'],
                "val_accuracy": history.history['val_accuracy']
            }
        }

        metrics_path = str(model_output_dir / 'performance_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print("Performance metrics and history saved.")
        return True, None
    
    except Exception as e:
        print(f"Saving performance metrics failed due to {e}")
        return False, e
