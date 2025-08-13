import datasets
import librosa
import numpy as np
import pathlib
import os
import json
import tensorflow as tf
from dotenv import load_dotenv
import random
load_dotenv()

dataset_path = pathlib.Path(os.getenv('dataset_path', './data'))
output_path = dataset_path / "spoken_digit_spectrograms"
tf_dataset_output_dir = dataset_path / "spoken_digit_tf_datasets"
config_path = pathlib.Path(os.getenv('config_file_path'))
output_path.mkdir(exist_ok=True)
tf_dataset_output_dir.mkdir(exist_ok=True)

try:
    with open(str(config_path), 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found.")
    exit()

def add_noise(audio_data, noise_factor=0.005):
    noise = np.random.randn(len(audio_data))
    augmented_data = audio_data + noise_factor * noise
    augmented_data = augmented_data.astype(type(audio_data[0]))
    return augmented_data

def shift_pitch(audio_data, sampling_rate, pitch_factor=2):
    return librosa.effects.pitch_shift(y=audio_data, sr=sampling_rate, n_steps=pitch_factor)

def create_spectrogram_and_augment(example):
    try:
        audio_data = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        target_sr = config['audio_params']['sample_rate']

        if sampling_rate != target_sr:
            audio_data = librosa.resample(y=audio_data, orig_sr=sampling_rate, target_sr=target_sr)
        if random.random() > 0.5: # 50% chance to add noise
            audio_data = add_noise(audio_data)
        if random.random() > 0.5: # 50% chance to shift pitch
            pitch_change = random.choice([-2, -1, 1, 2])
            audio_data = shift_pitch(audio_data, target_sr, pitch_change)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data,
            sr=target_sr,
            n_fft=config['audio_params']['n_fft'],
            hop_length=config['audio_params']['hop_length'],
            n_mels=config['audio_params']['n_mels']
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return {"mel_spectrogram": log_mel_spectrogram, "label": example["label"]}

    except Exception as e:
        print(f"Error processing a file: {e}")
        return {"mel_spectrogram": None, "label": None}

def convert_to_spectrogram():
    try:
        train_parquet_path = dataset_path / "train.parquet"
        if not train_parquet_path.exists():
            raise FileNotFoundError(f"Error: The file '{train_parquet_path}' was not found.")

        dataset = datasets.Dataset.from_parquet(str(train_parquet_path))
        dataset = dataset.filter(lambda example: example['audio'] is not None)

        spectrogram_dataset = dataset.map(
            create_spectrogram_and_augment,
            remove_columns=["audio"],
            num_proc=4
        )
        spectrogram_dataset = spectrogram_dataset.filter(lambda x: x['mel_spectrogram'] is not None)

        final_datasets = spectrogram_dataset.train_test_split(test_size=0.1, seed=42)
        final_datasets["validation"] = final_datasets.pop("test")

        for split, data in final_datasets.items():
            file_path = output_path / f"{split}.parquet"
            data.to_parquet(file_path)
        print(f"\nSaved augmented datasets as Parquet files to '{output_path}'")
        return True, None
    except Exception as e:
        print(f"An error occurred during spectrogram conversion: {e}")
        return False, e

def preprocess_for_training():
    print("Starting preprocessing for training...")
    try:
        for split in ["train", "validation"]:
            parquet_path = output_path / f"{split}.parquet"
            if not parquet_path.exists():
                print(f"Spectrogram file not found for split: {split}. Skipping.")
                continue

            hf_dataset = datasets.Dataset.from_parquet(str(parquet_path))

            def data_generator():
                for example in hf_dataset:
                    spec = np.asarray(example['mel_spectrogram']).T
                    if spec.shape[0] > config['input_shape'][0]:
                        spec = spec[:config['input_shape'][0], :]
                    else:
                        pad_width = config['input_shape'][0] - spec.shape[0]
                        spec = np.pad(spec, ((0, pad_width), (0, 0)), mode='constant')
                    
                    label = tf.keras.utils.to_categorical([example['label']], num_classes=config['num_classes'])[0]
                    yield spec, label

            output_signature = (
                tf.TensorSpec(shape=(config['input_shape'][0], config['input_shape'][1]), dtype=tf.float32),
                tf.TensorSpec(shape=(config['num_classes'],), dtype=tf.float32)
            )
            tf_dataset = tf.data.Dataset.from_generator(
                data_generator, output_signature=output_signature
            ).batch(config['training_params']['batch_size'])

            tf_dataset_path = tf_dataset_output_dir / split
            tf.data.Dataset.save(tf_dataset, str(tf_dataset_path))
            print(f"Successfully preprocessed and saved '{split}' dataset to '{tf_dataset_path}'")
        return True, None
    except Exception as e:
        print(f"An error occurred during preprocessing for training: {e}")
        return False, e

def preprocess_dataset(FORCE_PREPROCESS = False):
    if FORCE_PREPROCESS or not os.path.exists(tf_dataset_output_dir) or not os.listdir(tf_dataset_output_dir):
        
        success, error = convert_to_spectrogram()
        if not success:
            print(f"Spectrogram creation failed with error: {error}")
            return False, error
        else:
            success, error = preprocess_for_training()
            if not success:
                print(f"Preprocessing for training failed with error: {error}")
                return False, error
            else:
                print("\nAll preprocessing steps completed successfully!")
                return True, None
    else:
        print("Preprocessed files already exist, skipping")
        return True, None
