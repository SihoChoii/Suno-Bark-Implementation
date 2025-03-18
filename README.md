# 🔊 Bark Long-form Text-to-Audio Generator

A Python tool for generating long-form audio from text using Suno AI's [Bark](https://github.com/suno-ai/bark) text-to-audio model. This script automatically splits your text into optimal chunks, generates high-quality audio for each chunk, and seamlessly combines them into a single audio file.

## ✨ Features

- **Smart Text Chunking**: Splits text by word count while respecting sentence boundaries for natural-sounding transitions
- **Voice Consistency**: Uses the same voice preset across all chunks for a coherent listening experience
- **Customizable Transitions**: Adds configurable silence between chunks for improved pacing
- **Memory Efficient**: Option to use smaller models for devices with limited VRAM
- **PyTorch 2.6 Compatible**: Works with the latest PyTorch versions
- **ChatGPT Integration**: Automatically enhances your text with appropriate non-speech sounds and emphasis markers
- **Time Tracking**: Shows progress, elapsed time, and estimated completion time

## 🛠️ Installation

### Prerequisites

- Python 3.7+ installed on your system
- Git installed on your system
- An OpenAI API key (for ChatGPT integration)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/bark-longform-generator.git
cd bark-longform-generator
```

2. Create and activate a virtual environment:

**On macOS/Linux:**
```bash
# Create virtual environment
python -m venv bark-env

# Activate virtual environment
source bark-env/bin/activate
```

**On Windows:**
```bash
# Create virtual environment
python -m venv bark-env

# Activate virtual environment (Command Prompt)
bark-env\Scripts\activate.bat

# Activate virtual environment (PowerShell)
bark-env\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
# Install Bark from GitHub
pip install git+https://github.com/suno-ai/bark.git

# Install required packages
pip install numpy scipy torch huggingface_hub transformers librosa tqdm accelerate openai
```

> Note: For GPU support, you may need to install a specific PyTorch version compatible with your CUDA version. Visit [PyTorch's installation page](https://pytorch.org/get-started/locally/) for specific instructions.

4. Set your OpenAI API key (if using ChatGPT enhancement):

**On macOS/Linux:**
```bash
export OPENAI_API_KEY=your_api_key_here
```

**On Windows Command Prompt:**
```bash
set OPENAI_API_KEY=your_api_key_here
```

**On Windows PowerShell:**
```bash
$env:OPENAI_API_KEY="your_api_key_here"
```

## 📋 Usage

### Basic Usage

```bash
python bark_longform.py --input your_text_file.txt --output output_audio.wav
```

### With ChatGPT Enhancement

```bash
python bark_longform.py --input your_text_file.txt --output output_audio.wav --non_speech_weight 0.7
```

### Advanced Options

```bash
python bark_longform.py --input your_text_file.txt --output output_audio.wav --words_per_chunk 40 --voice_preset "v2/en_speaker_6" --silence_between_chunks 0.3 --non_speech_weight 0.5
```
The library of supported voice preset can be found here: https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input text file | (Required) |
| `--output` | Path to output audio file | (Required) |
| `--words_per_chunk` | Number of words per chunk | 40 |
| `--voice_preset` | Voice preset for Bark (e.g., "v2/en_speaker_1") | None (random voice) |
| `--silence_between_chunks` | Duration of silence between chunks in seconds | 0.5 |
| `--small_models` | Use small models to reduce VRAM usage | False |
| `--force_cpu` | Force CPU usage even if CUDA is available | False |
| `--non_speech_weight` | Weight (0-1) for adding non-speech sounds via ChatGPT | 0.0 |
| `--openai_api_key` | OpenAI API key (optional, can use env var instead) | None |

## 🎭 ChatGPT Enhancement

When using the `--non_speech_weight` parameter (value between 0 and 1), the script will send your text to ChatGPT for semantic analysis and enhancement with appropriate non-speech sounds and formatting:

- **0.0**: No enhancement, skips ChatGPT processing
- **0.1-0.3**: Minimal enhancement with sparse non-speech elements
- **0.4-0.7**: Moderate enhancement with balanced non-speech elements
- **0.8-1.0**: Maximum enhancement with abundant non-speech elements

Supported non-speech markers that Bark recognizes:

- `[laughter]` or `[laughs]` - For moments of humor
- `[sighs]` - For moments of resignation, relief, or exhaustion
- `[music]` - For moments that would benefit from musical emphasis
- `[gasps]` - For moments of surprise or shock
- `[clears throat]` - For appropriate moments
- `—` or `...` - For natural hesitations or pauses
- `♪` - To surround song lyrics
- `CAPITALIZATION` - For emphasis of specific words

An enhanced version of your text will be saved alongside the audio output for reference.

## 💻 Example

```python
from bark_longform import process_text_to_audio

# Generate audio from a text file with ChatGPT enhancement
process_text_to_audio(
    input_file="story.txt",
    output_file="story_audio.wav",
    words_per_chunk=40,
    voice_preset="v2/en_speaker_6",
    silence_between_chunks=0.3,
    non_speech_weight=0.7,
    openai_api_key="your_api_key_here"  # Optional if set as environment variable
)
```

## 🧠 How It Works

1. **Text Enhancement**: If enabled, the script sends your text to ChatGPT for semantic analysis to add appropriate non-speech sounds and formatting.

2. **Text Parsing**: The script reads your (potentially enhanced) text file and intelligently splits it into chunks of approximately the specified word count, respecting sentence boundaries.

3. **Audio Generation**: Each chunk is processed by Bark to generate high-quality audio. The same voice preset is used for all chunks to maintain consistency.

4. **Audio Combination**: The generated audio segments are combined with configurable silence between chunks to create a single, coherent audio file.

## ⚠️ Troubleshooting

### OpenAI API Issues

If you encounter errors with the ChatGPT enhancement:
- Ensure your API key is correct and has sufficient credits
- Check your internet connection
- If issues persist, set `--non_speech_weight 0` to skip the enhancement

### PyTorch 2.6 Loading Error

If you encounter an error like:
```
_pickle.UnpicklingError: Weights only load failed...
```

This is due to PyTorch 2.6's stricter security measures. Our script includes a fix for this issue. If you still encounter problems, you can try setting these environment variables before running:

```python
import os
os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
```

### CUDA Out of Memory

If you encounter GPU memory issues:
1. Use the `--small_models` flag to reduce VRAM usage
2. Reduce the `--words_per_chunk` value

## 🙏 Acknowledgments

This project is built on top of [Suno AI's Bark](https://github.com/suno-ai/bark), an impressive text-to-audio model. All credit for the underlying audio generation capabilities goes to their team.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📢 Disclaimer

Please use this tool responsibly and in accordance with Bark's usage guidelines. The tool is intended for research and creative purposes. Do not use it to create misleading content or to infringe on anyone's rights.