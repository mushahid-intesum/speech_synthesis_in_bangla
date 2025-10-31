<p align="center">
    <img src="resources/reverse-diffusion.gif" alt="drawing" width="500"/>
</p>


# STFT-GradTTS: A Robust, Diffusion-based Speech Synthesis System with iSTFT decoder for Bangla

Official implementation of A Robust Text-to-Speech System in Bangla with Stochastic
Duration Predictor

**Authors**: Mushahid Intesum\**, Abdullah Ibne Masud\**, Md Ashraful Islam\**, Dr Md Rezaul Karim.

<sup>\*Equal contribution.</sup>
<sup>\**Corresponding Author.</sup>

## Abstract

Text-to-speech (TTS) synthesis is a critical area in speech and language processing, with extensive
applications in assistive technologies, virtual assistants, and automated content generation. Despite
Bangla being the seventh most spoken language globally, high-quality TTS datasets remain scarce,
limiting advancements in Bangla speech synthesis. To bridge this gap, we introduce a meticulously
curated single-speaker Bangla audio dataset comprising over 20 hours of clean speech. Our dataset
ensures diverse linguistic coverage, incorporating compound letters, long vowels, Sanskrit words, and
varied sentence structures while maintaining phonetic balance.
In addition to the dataset, we propose STFT-GradTTS, a novel diffusion-based TTS model featuring a
Multi-Stream iSTFT decoder and a Stochastic Duration Predictor (SDP). The iSTFT-based decoder
synthesizes high-fidelity waveforms by decomposing signals into sub-bands using learnable synthesis
filters, enhancing spectral clarity. The SDP models phoneme duration distributions, capturing the
natural variability in speech pacing. Together, these innovations address key limitations of GradTTS,
particularly in duration accuracy and audio naturalness.
We validate our approach through blind subjective evaluations using Mean Opinion Score (MOS)
assessments and objective evaluations using Mel Cepstral Distortion (MCD) and ViSQOL scores,
demonstrating that STFT-GradTTS significantly outperforms the baseline GradTTS in speech qual-
ity, naturalness, and duration modeling. Our work not only establishes a new benchmark for Bangla
Text-to-Speech (TTS) systems but also provides a solid foundation for future research in low-resource
language speech synthesis.

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
3. Run script `inference.py`. Before running comment out which model you want to run inside the script. The relevant parts are mentioned:
    ```bash
    python inference.py
    ```
4.You will

## Training

1. Make filelists of your audio data like ones included into `resources/filelists` folder. Make a new folder names `data` and put audio files inside `wavs` and text files in `text` folders respectively.
2. Make a `metadata.txt` file that has audio file name and the corresponding text. An example is:
```
1233456|this is the text
``` 
3. Set experiment configuration in `params.py` file.
4. In the `train.py` script, comment out the relevant parts according to the instructions inside the script
5. Specify your GPU device and run training script:
    ```bash
    python train.py 
    ```
4. To track your training process run tensorboard server on any available port:
    ``` bash
    tensorboard --logdir=YOUR_LOG_DIR --port=8888
    ```
    During training all logging information and checkpoints are stored in `YOUR_LOG_DIR`, which you can specify in `params.py` before training.


### Note
If you'd like to run the SDP+Context model, simple comment out the parts under GradTTSSDP model comments and uncomment the parts under GradTTSSDPContext in `train.py` and `inference.py`
