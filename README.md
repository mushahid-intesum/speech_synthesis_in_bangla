<p align="center">
    <img src="resources/reverse-diffusion.gif" alt="drawing" width="500"/>
</p>


# A Robust Text-to-Speech System in Bangla with Stochastic Duration Predictor

Official implementation of A Robust Text-to-Speech System in Bangla with Stochastic
Duration Predictor

**Authors**: Mushahid Intesum\**, Abdullah Ibne Masud\**, Md Ashraful Islam\**, Dr Md Rezaul Karim.

<sup>\*Equal contribution.</sup>
<sup>\**Corresponding Author.</sup>

## Abstract

Text-to-speech (TTS), a field aiming to produce natural speech from text, is a prominent area of research in speech, language, and machine learning, with broad industrial applications. Despite Bangla being the seventh most spoken language globally, there exists a significant shortage of high-quality audio data for TTS, automatic speech recognition, and other audio-related natural language processing tasks. To address this gap, we have compiled a meticulously curated single-speaker Bangla audio dataset. Following extensive preprocessing, our dataset has more than 20 hours of clean audio data featuring a diverse array of genres and sources, supplemented by novel metrics, including categorization based on sentence complexity, distribution of tense and person, as well as quantitative measurements such as word count, unique word count, and compound letter count. Our dataset, along with its distinctive evaluation metrics, fills a significant void in the evaluation of Bangla audio datasets, rendering it a valuable asset for future research endeavors. Additionally, we propose a novel TTS model employing diffusion and a duration predictor. Our model integrates a Stochastic Duration Predictor(SDP) to enhance alignment between input text and speech duration, alongside a context prediction network for improved word pronunciation. The incorporation of the SDP aims to emulate the variability observed in human speech, where the same sentence may be pronounced with different duration. This addition facilitates the generation of more natural-sounding audio samples with improved duration characteristics. Through blind subjective analysis utilizing the Mean Opinion Score (MOS), we demonstrate that our proposed model enhances the quality of the state-of-the-art GradTTS model.

## Installation

Install all Python package requirements:

```bash
pip install -r requirements.txt
```

**Note**: code is tested on Python==3.10.9.

## Inference

You can download BnTTS dataset (22kHz) from [here](https://drive.google.com/drive/folders/195CgbUxViuGg0aJKUSvkYulTJEf3eUbS?usp=drive_link).

Put necessary HiFi-GAN checkpoints into `checkpts` folder in root directory (note: in `inference.py` you can change default HiFi-GAN path).

1. Create text file with sentences you want to synthesize like `resources/filelists/synthesis.txt`.
2. For single speaker set `params.n_spks=1`.
3. Run script `inference.py` by providing path to the text file, path to the model checkpoint, number of iterations to be used for reverse diffusion (default: 10) and speaker id if you want to perform multispeaker inference:
    ```bash
    python inference.py -f <your-text-file> -c <bn-tts-checkpoint> -t <number-of-timesteps> 
    ```
4. Check out folder called `out` for generated audios.

## Training

1. Make filelists of your audio data like ones included into `resources/filelists` folder. Make a new folder names `data` and put audio files inside `wavs` and text files in `text` folders respectively.
2. Make a `metadata.txt` file that has audio file name and the corresponding text. An example is:
```
1233456|this is the text
``` 
3. Set experiment configuration in `params.py` file.
4. Specify your GPU device and run training script:
    ```bash
    python train.py 
    ```
5. To track your training process run tensorboard server on any available port:
    ``` bash
    tensorboard --logdir=YOUR_LOG_DIR --port=8888
    ```
    During training all logging information and checkpoints are stored in `YOUR_LOG_DIR`, which you can specify in `params.py` before training.


### Note
If you'd like to run the SDP+Context model, simple comment out the parts under GradTTSSDP model comments and uncomment the parts under GradTTSSDPContext in `train.py` and `inference.py`
