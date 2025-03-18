import argparse
import os
import re
import time
import numpy as np
import torch
from scipy.io.wavfile import write as write_wav
from datetime import timedelta
from openai import OpenAI

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

def enhance_text_with_chatgpt(text, non_speech_weight, api_key=None):
    """
    Use ChatGPT to enhance text with non-speech sounds based on semantic analysis.
    
    Args:
        text: The input text to enhance
        non_speech_weight: A float between 0 and 1 indicating how much to add non-speech sounds
        api_key: OpenAI API key (optional, will use environment variable if not provided)
    
    Returns:
        Enhanced text with non-speech sounds
    """
    # Skip if weight is 0
    if non_speech_weight <= 0:
        return text
    
    # Set up OpenAI client
    try:
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            client = OpenAI()  # Uses OPENAI_API_KEY environment variable
    except Exception as e:
        print(f"Error setting up OpenAI client: {e}")
        print("Make sure you've set the OPENAI_API_KEY environment variable or provided an API key.")
        print("Proceeding with original text...")
        return text
    
    # Create prompt based on the weight parameter
    if non_speech_weight < 0.3:
        intensity = "sparingly and only when absolutely necessary"
    elif non_speech_weight < 0.7:
        intensity = "moderately where appropriate"
    else:
        intensity = "liberally throughout the text to make it as expressive as possible"
    
    prompt = f"""
    Analyze the following text semantically and enhance it with appropriate non-speech sounds and formatting for text-to-speech.
    
    Please add non-speech elements {intensity}. Use any of the following where contextually appropriate:
    
    - [laughter] or [laughs] for moments of humor
    - [sighs] for moments of resignation, relief, or exhaustion
    - [music] for moments that would benefit from musical emphasis
    - [gasps] for moments of surprise or shock
    - [clears throat] for appropriate moments
    - "—" or "..." for natural hesitations or pauses
    - ♪ to surround song lyrics
    - CAPITALIZE words that deserve emphasis
    
    Original text:
    {text}
    
    Enhanced text (maintain all original meaning and content, just add appropriate non-speech elements):
    """
    
    # Call ChatGPT API
    try:
        print("Calling ChatGPT API to enhance text with non-speech sounds...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Can be adjusted based on needs
            messages=[
                {"role": "system", "content": "You're an expert in enhancing text for natural-sounding text-to-speech with appropriate non-verbal elements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        enhanced_text = response.choices[0].message.content.strip()
        
        processing_time = time.time() - start_time
        print(f"ChatGPT API processing completed in {format_time(processing_time)}")
        
        return enhanced_text
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        print("Proceeding with original text...")
        return text

def format_time(seconds):
    """Format seconds into a human-readable time string."""
    return str(timedelta(seconds=round(seconds)))

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

def process_text_to_audio(input_file, output_file, words_per_chunk, voice_preset=None, 
                         silence_between_chunks=0.5, force_cpu=False, 
                         non_speech_weight=0.0, openai_api_key=None):
    """
    Process a text file into audio using Bark.
    
    Args:
        input_file: Path to the input text file
        output_file: Path to save the output audio file
        words_per_chunk: Number of words per chunk to process
        voice_preset: Optional voice preset for Bark
        silence_between_chunks: Duration of silence between chunks in seconds
        force_cpu: Force CPU usage even if CUDA is available
        non_speech_weight: Weight (0-1) for adding non-speech sounds via ChatGPT
        openai_api_key: OpenAI API key (optional)
    """
    # Check CUDA availability
    cuda_available = torch.cuda.is_available() and not force_cpu
    device = "GPU" if cuda_available else "CPU"
    
    print(f"Device: Using {device} for inference")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        print(f"GPU: {gpu_name} with {gpu_memory:.2f} GB VRAM")
    
    # Load models
    print("Loading Bark models...")
    start_time = time.time()
    preload_models()
    model_load_time = time.time() - start_time
    print(f"Models loaded in {format_time(model_load_time)}")
    
    # Read text file
    print(f"Reading text from {input_file}...")
    text = read_text_file(input_file)
    
    # Enhance text with ChatGPT if non_speech_weight > 0
    if non_speech_weight > 0:
        print(f"Enhancing text with non-speech sounds (weight: {non_speech_weight})...")
        enhanced_text = enhance_text_with_chatgpt(text, non_speech_weight, openai_api_key)
        
        # Save the enhanced text to a file for reference
        enhanced_file = f"{os.path.splitext(output_file)[0]}_enhanced.txt"
        with open(enhanced_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_text)
        print(f"Enhanced text saved to {enhanced_file}")
        
        # Use enhanced text for audio generation
        text = enhanced_text
    
    # Split text into chunks
    print(f"Splitting text into chunks of approximately {words_per_chunk} words...")
    chunks = split_into_chunks(text, words_per_chunk)
    print(f"Created {len(chunks)} chunks")
    
    # Generate audio for each chunk
    print("Generating audio for each chunk...")
    audio_arrays = []
    
    total_start_time = time.time()
    estimated_total_time = None
    
    for i, chunk in enumerate(chunks):
        chunk_start_time = time.time()
        print(f"\nProcessing chunk {i+1}/{len(chunks)}:")
        print(f"Chunk text: {chunk[:100]}..." if len(chunk) > 100 else f"Chunk text: {chunk}")
        
        # Generate audio using consistent voice preset
        audio_array = generate_audio(chunk, history_prompt=voice_preset)
        
        # Add silence between chunks (except for the last chunk)
        if i < len(chunks) - 1:
            audio_array = add_silence(audio_array, silence_between_chunks)
            
        audio_arrays.append(audio_array)
        
        # Calculate and print timing information
        chunk_processing_time = time.time() - chunk_start_time
        elapsed_time = time.time() - total_start_time
        chunks_remaining = len(chunks) - (i + 1)
        
        # Calculate estimated time after the first chunk
        if i == 0 and len(chunks) > 1:
            estimated_total_time = chunk_processing_time * len(chunks)
        
        print(f"Chunk {i+1} completed in {format_time(chunk_processing_time)}")
        print(f"Elapsed time: {format_time(elapsed_time)}")
        
        if chunks_remaining > 0 and estimated_total_time is not None:
            estimated_remaining = (estimated_total_time - elapsed_time)
            # Adjust the estimate based on actual processing time of completed chunks
            if i > 0:
                avg_chunk_time = elapsed_time / (i + 1)
                estimated_remaining = avg_chunk_time * chunks_remaining
            
            print(f"Estimated time remaining: {format_time(estimated_remaining)}")
            print(f"Estimated completion at: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_remaining))}")
    
    # Combine audio arrays
    print("\nCombining audio chunks...")
    combined_audio = np.concatenate(audio_arrays)
    
    # Save to file
    print(f"Saving audio to {output_file}...")
    write_wav(output_file, SAMPLE_RATE, combined_audio)
    
    # Print summary
    total_time = time.time() - total_start_time
    print(f"\nDone! Audio saved to {output_file}")
    print(f"Total processing time: {format_time(total_time)}")
    print(f"Average time per chunk: {format_time(total_time / len(chunks))}")
    
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
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')
    parser.add_argument('--non_speech_weight', type=float, default=0.0,
                        help='Weight (0-1) for adding non-speech sounds via ChatGPT (default: 0.0)')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key (optional, will use OPENAI_API_KEY env var if not provided)')
    
    args = parser.parse_args()
    
    # Validate non_speech_weight
    if args.non_speech_weight < 0 or args.non_speech_weight > 1:
        print("Error: non_speech_weight must be between 0 and 1")
        return
    
    # Set environment variables for small models if requested
    if args.small_models:
        os.environ["SUNO_USE_SMALL_MODELS"] = "True"
    
    process_text_to_audio(
        args.input, 
        args.output, 
        args.words_per_chunk, 
        args.voice_preset,
        args.silence_between_chunks,
        args.force_cpu,
        args.non_speech_weight,
        args.openai_api_key
    )

if __name__ == "__main__":
    main()