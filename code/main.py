from download_dataset import download_dataset
from preprocess_audio_input import preprocess_dataset
from train import train
from predict import evaluate_on_test_set
from predict import prediction
import traceback
import os
from dotenv import load_dotenv
load_dotenv()

forcedownload = os.getenv('FORCE_DOWNLOAD').lower() in ('true', '1', 't', 'yes', 'y')
forcepreprocess = forcedownload or os.getenv('FORCE_PREPROCESS').lower() in ('true', '1', 't', 'yes', 'y')
forcetrain = forcepreprocess or os.getenv('FORCE_TRAIN').lower() in ('true', '1', 't', 'yes', 'y')

if __name__ == "__main__":
    status , e = download_dataset(FORCE_DOWNLOAD=forcedownload)
    if status:
        status, e = preprocess_dataset(FORCE_PREPROCESS=forcepreprocess)
        if status:
            status , e = train(FORCE_TRAIN=forcetrain)
            if status:
                evaluate_on_test_set()
                prediction()
            else:
                traceback.print_exception(e)
        else:
            traceback.print_exception(e)
    else:
        traceback.print_exception(e)