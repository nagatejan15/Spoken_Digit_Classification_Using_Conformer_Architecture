import tensorflow as tf
import librosa
import numpy as np
import datasets
import sounddevice as sd
from scipy.io.wavfile import write
import json
import pathlib
import os
from dotenv import load_dotenv
from scipy import stats
load_dotenv()

model_output_dir = pathlib.Path(os.getenv('model_path'))
config_path = pathlib.Path(os.getenv('config_file_path'))
dataset_dir = pathlib.Path(os.getenv('dataset_path'))

with open(str(config_path), 'r') as f:
    config = json.load(f)
try:
    model_save_path = model_output_dir / f"{config['model_name']}.keras"
    model = tf.keras.models.load_model(str(model_save_path))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

def preprocess_for_evaluation(audio_array, sr, config):
    
    audio_params = config.get('audio_params', {})
    target_sr = audio_params.get('sample_rate', 8000)
    n_fft = audio_params.get('n_fft', 256)
    hop_length = audio_params.get('hop_length', 80)
    n_mels = audio_params.get('n_mels', 40)
    max_len = config['input_shape'][0]

    if sr != target_sr:
        audio_array = librosa.resample(y=np.float32(audio_array), orig_sr=sr, target_sr=target_sr)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_array, sr=target_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    spec = log_mel_spectrogram.T
    if spec.shape[0] > max_len:
        spec = spec[:max_len, :]
    else:
        pad_width = max_len - spec.shape[0]
        spec = np.pad(spec, ((0, pad_width), (0, 0)), mode='constant')
        
    return tf.expand_dims(spec, axis=0)

def preprocess_for_prediction(audio_array, sr, config):
    
    audio_params = config.get('audio_params', {})
    target_sr = audio_params.get('sample_rate', 8000)
    n_fft = audio_params.get('n_fft', 256)
    hop_length = audio_params.get('hop_length', 80)
    n_mels = audio_params.get('n_mels', 40)
    max_len = config['input_shape'][0]

    if sr != target_sr:
        audio_array = librosa.resample(y=np.float32(audio_array), orig_sr=sr, target_sr=target_sr)
        
    augmented_audio = [audio_array] 
    # Pitch shifting
    augmented_audio.append(librosa.effects.pitch_shift(y=audio_array, sr=target_sr, n_steps=2))
    augmented_audio.append(librosa.effects.pitch_shift(y=audio_array, sr=target_sr, n_steps=-2))
    # Noise injection
    noise = np.random.randn(len(audio_array))
    noisy_audio = audio_array + 0.005 * noise
    augmented_audio.append(noisy_audio.astype(type(audio_array[0])))


    processed_tensors = []
    for audio in augmented_audio:
        audio, _ = librosa.effects.trim(audio, top_db=20)
        audio = librosa.effects.preemphasis(audio)
        noise_clip = audio[:int(target_sr*0.1)]
        stft_noise = librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop_length)
        noise_profile = np.mean(np.abs(stft_noise), axis=1)
        stft_signal = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        stft_signal_abs = np.abs(stft_signal)
        stft_denoised_abs = np.maximum(0, stft_signal_abs - noise_profile[:, np.newaxis])
        audio = librosa.istft(stft_denoised_abs * (stft_signal / (stft_signal_abs + 1e-10)), hop_length=hop_length)

        # Spectrogram creation
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=target_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Final padding and shaping
        spec = log_mel_spectrogram.T
        if spec.shape[0] > max_len:
            spec = spec[:max_len, :]
        else:
            pad_width = max_len - spec.shape[0]
            spec = np.pad(spec, ((0, pad_width), (0, 0)), mode='constant')
            
        processed_tensors.append(tf.expand_dims(spec, axis=0))
        
    return processed_tensors


def predict_digit(model, processed_audio_tensors):
   
    predictions = []
    for tensor in processed_audio_tensors:
        prediction = model.predict(tensor, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        predictions.append(predicted_digit)
    
    # Return the most frequent prediction (the mode)
    final_prediction = stats.mode(predictions)[0]
    print(predictions)
    return final_prediction

def evaluate_on_test_set():
    print("\n Evaluating on Test Set ")
    try:
        test_file_path = dataset_dir / "test.parquet"
        hf_dataset = datasets.Dataset.from_parquet(str(test_file_path))
    except FileNotFoundError:
        print(f"Error: Test data file not found at '{test_file_path}'")
        return

    def data_generator():
        for example in hf_dataset:
            audio_array = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]
            label = example["label"]
            
            processed_tensor = preprocess_for_evaluation(audio_array, sr, config)
            spec = tf.squeeze(processed_tensor, axis=0) 
            one_hot_label = tf.keras.utils.to_categorical([label], num_classes=config['num_classes'])[0]
            yield spec, one_hot_label

    output_signature = (
        tf.TensorSpec(shape=(config['input_shape'][0], config['input_shape'][1]), dtype=tf.float32),
        tf.TensorSpec(shape=(config['num_classes'],), dtype=tf.float32)
    )
    test_tf_dataset = tf.data.Dataset.from_generator(
        data_generator, output_signature=output_signature
    ).batch(config['training_params']['batch_size'])

    loss, accuracy = model.evaluate(test_tf_dataset, verbose=1)
    print(f"\n Evaluation on Testdataset is Complete ")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    metrics_path = str(model_output_dir / 'performance_metrics.json')
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}

    metrics["test_metrics"] = {
        "test_loss": loss,
        "test_accuracy": accuracy
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Test metrics saved.")

def predict_from_file(model, config, file_path):
    try:
        audio_array, sr = librosa.load(file_path, sr=None)
        processed_tensors = preprocess_for_prediction(audio_array, sr, config)
        predicted_digit = predict_digit(model, processed_tensors)
        print(f"The predicted digit is: {predicted_digit}")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

def record_and_predict(model, config, duration=2, filename="recorded_digit.wav"):
    print("\n Record Audio from Microphone ")
    target_sr = config.get('audio_params', {}).get('sample_rate', 8000)
    print(f"Recording for {duration} seconds, Speak a digit now!")
    
    recording = sd.rec(int(duration * target_sr), samplerate=target_sr, channels=1, dtype='float32')
    sd.wait()
    
    write(filename, target_sr, recording)
    print(f"Recording saved as '{filename}'")
    
    predict_from_file(model, config, filename)

def prediction():
    try:
        while True:
            print("\nWhat would you like to do next?")
            print("1: Predict from an audio file")
            print("2: Record audio and predict")
            print("3: Exit")
            choice = input("Enter your choice (1-3): ")

            if choice == '1':
                file_path = input("Enter the path to the audio file: ")
                predict_from_file(model, config, file_path)
            elif choice == '2':
                record_and_predict(model, config)
            elif choice == '3':
                break
            else:
                print("Invalid choice. Please try again.")

    except (ValueError, FileNotFoundError) as e:
        print(f"\nError: {e}")
    except NameError:
        print("\nCould not start the application because the model or config failed to load.")
    except KeyboardInterrupt:
        print("\nReal-time prediction stopped by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
