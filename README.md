# 🔊 Bark Long-form Text-to-Audio Generator

A Python tool for generating long-form audio from text using Suno AI's [Bark](https://github.com/suno-ai/bark) text-to-audio model. This script automatically splits your text into optimal chunks, generates high-quality audio for each chunk, and seamlessly combines them into a single audio file.

## ✨ Features

- **Smart Text Chunking**: Splits text by word count while respecting sentence boundaries for natural-sounding transitions
- **Voice Consistency**: Uses the same voice preset across all chunks for a coherent listening experience
- **Customizable Transitions**: Adds configurable silence between chunks for improved pacing
- **Memory Efficient**: Option to use smaller models for devices with limited VRAM
- **PyTorch 2.6 Compatible**: Works with the latest PyTorch versions

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- PyTorch 2.0+
- Bark dependencies

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/bark-longform-generator.git
cd bark-longform-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually install the required packages:
```bash
pip install git+https://github.com/suno-ai/bark.git
pip install numpy scipy
```

## 📋 Usage

### Basic Usage

```bash
python bark_longform.py --input your_text_file.txt --output output_audio.wav
```

### Advanced Options

```bash
python bark_longform.py --input your_text_file.txt --output output_audio.wav --words_per_chunk 40 --voice_preset "v2/en_speaker_6" --silence_between_chunks 0.3 --small_models
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input text file | (Required) |
| `--output` | Path to output audio file | (Required) |
| `--words_per_chunk` | Number of words per chunk | 40 |
| `--voice_preset` | Voice preset for Bark (e.g., "v2/en_speaker_1") | None (random voice) |
| `--silence_between_chunks` | Duration of silence between chunks in seconds | 0.5 |
| `--small_models` | Use small models to reduce VRAM usage | False |

### Voice Presets

Bark supports [100+ speaker presets](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) across multiple languages. Some popular English presets include:

- `v2/en_speaker_1` - Male voice
- `v2/en_speaker_6` - Female voice 
- `v2/en_speaker_9` - Deep male voice

## 💻 Example

```python
from bark_longform import process_text_to_audio

# Generate audio from a text file
process_text_to_audio(
    input_file="story.txt",
    output_file="story_audio.wav",
    words_per_chunk=40,
    voice_preset="v2/en_speaker_6",
    silence_between_chunks=0.3
)
```

## 🧠 How It Works

1. **Text Parsing**: The script reads your text file and intelligently splits it into chunks of approximately the specified word count, respecting sentence boundaries.

2. **Audio Generation**: Each chunk is processed by Bark to generate high-quality audio. The same voice preset is used for all chunks to maintain consistency.

3. **Audio Combination**: The generated audio segments are combined with configurable silence between chunks to create a single, coherent audio file.

## ⚠️ Troubleshooting

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