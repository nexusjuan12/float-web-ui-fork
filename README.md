# FLOAT Web UI: Enhanced Interface for Audio-driven Talking Portrait

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

**Enhanced Web Interface** built upon the official [FLOAT](https://deepbrainai-research.github.io/float/) implementation

![preview](./demo-web-ui.png)

## ✨ Enhanced Features
Developed from the original [FLOAT](https://github.com/DeepBrainAI-Research/FLOAT) research implementation with significant usability improvements:

1. **Intuitive Web Interface**  
   - Drag-and-drop file uploads
   - Real-time previews
   - One-click generation

2. **Simplified Workflow**  
   - Automatic file handling
   - Clean output management
   - Progress indicators

3. **Extended Accessibility**  
   - Public sharing option (`--share`)
   - Custom server configuration (`--port`, `--server`)
   - Mobile-friendly design

## Quick Start

### Installation
```bash
# 1. Create Conda Environment
conda create -n float-web-ui python=3.8.5
conda activate float-web-ui

# 2. Install torch and requirements
sh environments.sh

# or manual installation
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
- Test on Linux, 4060 ti 16GB VRAM , RAM 32GB 

### Preparing checkpoints

1. Download checkpints automatically

    ```.bash
    sh download_checkpoints.sh
    ```

    or download checkpoints manually from this [google-drive](https://drive.google.com/file/d/1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0/view?usp=sharing).

2. The checkpoints should be organized as follows:
    ```.bash
    ./checkpints
    |-- checkpoints_here
    |-- float.pth                                       # main model
    |-- wav2vec2-base-960h/                             # audio encoder
    |   |-- .gitattributes
    |   |-- config.json
    |   |-- feature_extractor_config.json
    |   |-- model.safetensors
    |   |-- preprocessor_config.json
    |   |-- pytorch_model.bin
    |   |-- README.md
    |   |-- special_tokens_map.json
    |   |-- tf_model.h5
    |   |-- tokenizer_config.json
    |   '-- vocab.json
    '-- wav2vec-english-speech-emotion-recognition/     # emotion encoder
        |-- .gitattributes
        |-- config.json
        |-- preprocessor_config.json
        |-- pytorch_model.bin
        |-- README.md
        '-- training_args.bin
    ```
   - W2V based models could be found in the links: [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) and [wav2vec-english-speech-emotion-recognition](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition).

### Basic Usage
1. **Prepare Inputs**:
   - Image: Front-facing portrait (512×512 recommended)
   - Audio: Clean speech (WAV format, 16kHz recommended)

2. **Web Interface**:
   ```bash
   python app.py --port 7860 --share
   Drag & drop your files
   Select emotion and intensity
   Click "Generate"
   ```

Command Line:
```.bash
python generate.py \
    --ref_path image.png \
    --aud_path audio.wav \
    --emo happy \
    --e_cfg_scale 5
Advanced Options
Parameter	Description	Recommended
--a_cfg_scale	Audio influence (1-10)	2-3
--e_cfg_scale	Emotion intensity (1-10)	5-7 for strong effect
--no_crop	Disable auto-face-crop	Only for pre-cropped images
--seed	Random seed	15-100
```

Pro Tips :
   Use --emo neutral for subtle lip-sync only
   For musical audio, extract vocals first
   Higher --e_cfg_scale values (8-10) create dramatic expressions
