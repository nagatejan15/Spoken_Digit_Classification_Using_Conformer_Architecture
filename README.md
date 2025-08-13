# **Spoken Digit Classification using a Conformer Architecture Model**

## **Overview**

This project implements a complete pipeline to train and evaluate a **Conformer-based neural network** for classifying spoken digits (0-9). The model is trained on the [mteb/free-spoken-digit-dataset](https://huggingface.co/datasets/mteb/free-spoken-digit-dataset) from Hugging Face.

To improve the model's robustness against real-world audio variations, the project incorporates two key techniques:

1. **Data Augmentation:** During preprocessing, the training data is augmented with random **noise injection** and **pitch shifting**. This creates a more diverse and realistic dataset, helping the model generalize better.  
2. **Test-Time Augmentation (TTA):** For single audio predictions (from a file or microphone), the model uses TTA. It creates multiple altered versions of the input audio, gets a prediction for each, and returns the most common result (majority vote) for a more reliable final prediction.

The entire process is managed by a main script that handles data downloading, preprocessing, training, and evaluation.

## **Project Structure**

The project is organized into several Python scripts, each with a specific responsibility:

* main.py: The main entry point to run the entire pipeline sequentially.  
* download\_dataset.py: Downloads the raw audio dataset from Hugging Face.  
* preprocess\_audio\_input.py: Converts raw audio into spectrograms, applies data augmentation, and saves the data in a format ready for TensorFlow.  
* train.py: Builds the Conformer model architecture, trains it on the preprocessed data, and saves the trained model and performance metrics.  
* predict.py: Contains functions to evaluate the trained model on the test set and to make predictions on new audio from a file or a microphone recording.  
* .env: Configuration file for setting up file paths and execution flags.  
* config.json: A JSON file that defines the model's architecture, and audio and training parameters.  
* requirements.txt: A list of all necessary Python libraries for the project.

## **Setup and Installation**

Follow these steps to set up the project environment.

### **1\. Install Dependencies**

Install all required libraries using the requirements.txt file.
```
pip install -r requirements.txt
```

### **2\. Configure Environment Variables**

Create a .env file in the root of the project directory. This file manages file paths and execution flags.

**.env file template:**
```
# File Paths  
dataset_path=./data  
model_path=./model  
config_file_path=./model/config.json

# Execution Flags (set to True to force a step)  
FORCE_DOWNLOAD=False  
FORCE_PREPROCESS=False  
FORCE_TRAIN=False
```
### **3\. Create the Model Configuration File**

Create a config.json file in the root of the project directory. This file defines the entire model structure and its parameters.

## **How to Run the Project**

The project is designed to be run from a single entry point.

### **Execute the Full Pipeline**

To run the entire process (download, preprocess, train, evaluate, and predict), simply execute the main.py script:
```
python main.py
```
The script will perform the following steps in order:

1. **Download Dataset:** It will check if the dataset exists. If not, it will download it.  
2. **Preprocess Data:** It will check if the preprocessed data exists. If not, it will generate augmented spectrograms.  
3. **Train Model:** It will check if a trained model exists. If not, it will build and train the Conformer model.  
4. **Evaluate Model:** It will evaluate the trained model on the clean test set and save the metrics.  
5. **Start Prediction Menu:** It will launch an interactive menu allowing you to make predictions on new audio.

### **Forcing Steps with Flags**

By default, the script will skip steps if the output files already exist. You can force these steps to run again by setting the corresponding flags to True in your .env file.

* FORCE\_DOWNLOAD=True: Forces the dataset to be re-downloaded.  
* FORCE\_PREPROCESS=True: Forces the data to be re-processed and augmented.  
* FORCE\_TRAIN=True: Forces the model to be retrained, overwriting the existing model file.

## **How It Works**

### **Preprocessing and Augmentation**

The preprocess\_audio\_input.py script is responsible for preparing the data. For each audio file in the training set, it has a 50% chance of applying each of the following augmentations:

* **Noise Injection:** Adds a small amount of random noise to simulate background interference.  
* **Pitch Shifting:** Randomly shifts the pitch up or down to simulate different vocal tones.

This process creates a more robust training set that prepares the model for real-world audio.

### **Model Training**

The train.py script builds the Conformer model based on the config.json file. It then trains the model on the augmented spectrograms. Upon completion, it saves three important files in the directory specified by model\_path:

* conformer\_spoken\_digit\_classifier.keras: The final trained model.  
* performance\_metrics.json: A JSON file containing the training history and final validation metrics.  
* model\_architecture.png: A visual diagram of the model's layers.

### **Evaluation and Prediction**

After training, the predict.py script takes over.

* **Evaluation:** It first evaluates the model on the original, *clean* test dataset. This provides a reliable benchmark of the model's performance. The results are appended to the performance\_metrics.json file.  
* **Prediction:** It then presents an interactive menu. When you provide an audio file or record from your microphone, it uses **Test-Time Augmentation (TTA)**. It creates four versions of your input (original, pitch up, pitch down and noice) and selects the mode from four predictions.


## **Performance**
  
  **Training_Accuracy:** 0.9806584119796753
  
  **Validation_Accuracy:** 0.9333333373069763
  
  **Test_Accuracy:** 0.9866666793823242

  
  **Training_Loss:** 0.05561159923672676
  
  **Validation_Loss:** 0.29836907982826233
  
  **Test_Loss:** 0.056688882410526276
