import os
import random
from pathlib import Path
import shutil

def create_dataset_splits(wav_dir, text_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Create train, validation and test splits from audio and text files.
    
    Parameters:
    wav_dir (str): Directory containing wav files
    text_dir (str): Directory containing text files
    output_dir (str): Directory to save the split files
    train_ratio (float): Ratio of training data
    val_ratio (float): Ratio of validation data
    test_ratio (float): Ratio of test data
    """
    # Ensure ratios sum to 1
    # assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of wav files and their corresponding text files
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    dataset = []
    
    for wav_file in wav_files:
        # Get the corresponding text file
        base_name = os.path.splitext(wav_file)[0]
        text_file = os.path.join(text_dir, f"{base_name}.txt")
        
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                try:
                    text = f.read().strip()
                    dataset.append((base_name, text))
                except UnicodeDecodeError as e:
                    print(f'error in file {text_file}')
                    print(e)
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Calculate split indices
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Split the dataset
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    # Write splits to files
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = os.path.join(output_dir, f"{split_name}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for audio_id, text in split_data:
                f.write(f"{audio_id}|{text}\n")
        print(f"Created {split_name} set with {len(split_data)} samples")

if __name__ == "__main__":
    # Example usage
    wav_dir = "/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/resources/data/wavs"
    text_dir = "/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/resources/data/text"
    output_dir = "/mnt/Stuff/TTS/speech_synthesis_in_bangla-master/resources/data"
    
    create_dataset_splits(
        wav_dir=wav_dir,
        text_dir=text_dir,
        output_dir=output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )