import argparse
import os
import re
import numpy as np
import torch
from scipy.io.wavfile import write as write_wav

# Add safe global for PyTorch 2.6 compatibility
try:
    import numpy.core.multiarray
    from torch.serialization import add_safe_globals
    add_safe_globals([numpy.core.multiarray.scalar])
except (ImportError, AttributeError):
    pass  # Older PyTorch versions don't have add_safe_globals

# Monkey patch torch.load for Bark compatibility
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Now import Bark after patching torch.load
from bark import SAMPLE_RATE, generate_audio, preload_models

def read_text_file(file_path):
    """Read text from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_chunks(text, words_per_chunk):
    """
    Split text into chunks of approximately the target word count,
    trying to respect sentence boundaries when possible.
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        # If adding this sentence would exceed the limit and we already have content,
        # finalize the current chunk and start a new one
        if current_word_count + sentence_word_count > words_per_chunk and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0
        
        # Add the sentence to the current chunk
        current_chunk.append(sentence)
        current_word_count += sentence_word_count
        
        # If we've reached or exceeded the target with this addition, finalize the chunk
        if current_word_count >= words_per_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0
    
    # Add any remaining content as the final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def add_silence(audio_array, silence_duration=0.5):
    """Add silence to the end of an audio array."""
    silence_samples = int(silence_duration * SAMPLE_RATE)
    silence = np.zeros(silence_samples, dtype=np.float32)
    return np.concatenate([audio_array, silence])

def process_text_to_audio(input_file, output_file, words_per_chunk, voice_preset=None, silence_between_chunks=0.5):
    """
    Process a text file into audio using Bark.
    
    Args:
        input_file: Path to the input text file
        output_file: Path to save the output audio file
        words_per_chunk: Number of words per chunk to process
        voice_preset: Optional voice preset for Bark
        silence_between_chunks: Duration of silence between chunks in seconds
    """
    # Load models
    print("Loading Bark models...")
    preload_models()
    
    # Read text file
    print(f"Reading text from {input_file}...")
    text = read_text_file(input_file)
    
    # Split text into chunks
    print(f"Splitting text into chunks of approximately {words_per_chunk} words...")
    chunks = split_into_chunks(text, words_per_chunk)
    print(f"Created {len(chunks)} chunks")
    
    # Generate audio for each chunk
    print("Generating audio for each chunk...")
    audio_arrays = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}:")
        print(f"Chunk text: {chunk[:100]}..." if len(chunk) > 100 else f"Chunk text: {chunk}")
        
        # Generate audio using consistent voice preset
        audio_array = generate_audio(chunk, history_prompt=voice_preset)
        
        # Add silence between chunks (except for the last chunk)
        if i < len(chunks) - 1:
            audio_array = add_silence(audio_array, silence_between_chunks)
            
        audio_arrays.append(audio_array)
    
    # Combine audio arrays
    print("Combining audio chunks...")
    combined_audio = np.concatenate(audio_arrays)
    
    # Save to file
    print(f"Saving audio to {output_file}...")
    write_wav(output_file, SAMPLE_RATE, combined_audio)
    
    print(f"Done! Audio saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Convert text file to audio using Bark')
    parser.add_argument('--input', type=str, required=True, help='Path to input text file')
    parser.add_argument('--output', type=str, required=True, help='Path to output audio file')
    parser.add_argument('--words_per_chunk', type=int, default=40, 
                        help='Number of words per chunk (default: 40)')
    parser.add_argument('--voice_preset', type=str, default=None,
                        help='Voice preset for Bark (e.g., "v2/en_speaker_1")')
    parser.add_argument('--silence_between_chunks', type=float, default=0.5,
                        help='Duration of silence between chunks in seconds (default: 0.5)')
    parser.add_argument('--small_models', action='store_true',
                        help='Use small models to reduce VRAM usage')
    
    args = parser.parse_args()
    
    # Set environment variables for small models if requested
    if args.small_models:
        os.environ["SUNO_USE_SMALL_MODELS"] = "True"
    
    process_text_to_audio(
        args.input, 
        args.output, 
        args.words_per_chunk, 
        args.voice_preset,
        args.silence_between_chunks
    )

if __name__ == "__main__":
    main()