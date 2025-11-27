# Bose QA System

A custom Question-Answering system built using a fine-tuned Small Language Model (SLM) on DesignMax DM8SE documentation. The model used is Phi-3.5-mini-instruct as the base model, fine-tuned with QLoRA, and deployed via Ollama with a Gradio-based chat interface.


---
## Github link:
- https://github.com/Kapish-Rathod/bose-QA-system

---
## Model Link: 
- https://drive.google.com/drive/folders/1-kYE9csxhuRItvokZ8rF0ywNeHyn5XPm?usp=sharing
---
## Project Structure

```
bose-QA-system/
├── colab/                    
│   ├── DM8SE_SLM_Training.ipynb
│   └── Merge_model.ipynb
├── data/                     
│   ├── raw/                  
│   └── training/             
├── models/                   
│   └── downloaded/           
├── ui/                       
│   └── app.py               
├── docs/                    
├── requirements.txt         
└── README.md                
```

---

## Architecture

### System Overview

The Bose QA System follows a three-tier architecture:

```
┌─────────────────┐
│   Gradio UI     │  (Frontend - Chat Interface)
│   (Port 7860)   │
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│  Python Agent   │  (Middleware - API Client)
│   (app.py)      │
└────────┬────────┘
         │ Streaming API
         ▼
┌─────────────────┐
│   Ollama API    │  (Backend - Model Inference)
│  (Port 11434)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Bose SLM Model │  (Fine-tuned Phi-3.5-mini)
│   (GGUF Format) │
└─────────────────┘
```

### Model Architecture

**Base Model**: Microsoft Phi-3.5-mini-instruct
- **Architecture**: Phi3ForCausalLM
- **Parameters**: ~3.8B parameters
- **Hidden Size**: 3072
- **Number of Layers**: 32
- **Attention Heads**: 32
- **Max Position Embeddings**: 131,072 (with LongRoPE scaling)
- **Vocabulary Size**: 32,064
- **Activation Function**: SiLU
- **Normalization**: RMSNorm

### Data Preparation

Before fine-tuning the model, the training data was prepared through the following process:

1. **Raw Data Collection**:
   - The `tds_DesignMax_DM8SE_a4_EN.pdf` specification manual was downloaded (provided in the problem statement)
   - This PDF document serves as the raw data source containing product specifications and technical details

2. **Data Generation**:
   - The PDF content was provided to Gemini (Google's AI model) to generate question-answer pairs
   - Approximately 200 instruction-response pairs were generated from the specification manual

3. **Training Data Formatting**:
   - The generated text was formatted and input into `train.jsonl` file
   - The JSONL format contains instruction-response pairs suitable for supervised fine-tuning

### Fine-Tuning Architecture

The model is fine-tuned using **QLoRA (Quantized LoRA)**:

- **Quantization**: 4-bit quantization using BitsAndBytes
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
- **Training**: Supervised Fine-Tuning (SFT) was done with TRL library

### Deployment Architecture

1. **Training Phase** (Google Colab):
   - Model training with QLoRA adapters
   - Adapter weights were saved separately

2. **Merging Phase**:
   - LoRA adapters were merged with base model weights
   - Full model was saved in fp.16 format

3. **Conversion Phase**:
   - Model was converted to GGUF format using Ollama.cpp

4. **Serving Phase** (Ollama):
   - Model was loaded ussing Modelfile
   - Ollama API was used for making the model available using Gradio

5. **Interface Phase** (Gradio):
   - The chat UI connects to Ollama API

---

## Tools

### Training Tools

- **Transformers** (Hugging Face), **PEFT**, **TRL (Transformer Reinforcement Learning)**: (SFTTrainer), **BitsAndBytes**, **Accelerate**, **Datasets**

### Deployment Tools

- **Ollama**: Locally hostinf the model
- **Ollama.cpp**: Model conversion to GGUF format
- **Python Requests**: HTTP client for API communication

### UI Tools

- **Gradio**: Web-based chat interface framework
- **Python**: Core application logic

### Data Processing Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations


---

## APIs

### Ollama API

The system interacts with Ollama's REST API for model inference:

**Endpoint**: `http://localhost:11434/api/generate`

### Gradio Interface

The Gradio chat interface provides:
- **Input**: User prompt and conversation history
- **Output**: Text output by "Phi-3.5 mini" model

---

## Algorithms

### QLoRA (Quantized LoRA)

**Reason**: It works well in fine-tuning with reduced memory footprint

**Process**:
1. Base model is quantized to 4-bit precision
2. **LoRA Adapters**: Low-rank matrices added to attention and MLP layers
3. Only adapter weights are updated during fine-tuning
4. Adapters were merged back into base model after training

**Why**:
- Very les memory usage
- Model performance is good

### Supervised Fine-Tuning (SFT)

**Training Process**:
1. Training datat is in "instruction" and "response" pairs
4. Backpropagate through LoRA adapters

### Optimization Algorithms

- **Optimizer**: Paged AdamW 8-bit
  - Uses less memeory than AdamW 
- **Learning Rate Schedule**: 
  - Starts at initial learning rate
  - dynamic learning rate adjustment 


---

## Design Decisions

### 1. Model Selection: Phi-3.5-mini-instruct

**Why this model**:
- small model size (~3.8B parameters). easy for local deployment
- strong instruction-following capabilities
- good balance between performance and size
- other model like qwen are not as good performance wise for this specific application



### 2. QLoRA Fine-Tuning

**Why**:
- Less h/w requirements (Google Colab free tier)
- requires less training time compared to LoRa



### 3. Ollama Deployment

**WHy**:
- fast local deployment 
- No API 
- runs on my gpu-less laptop with GGUF format



### 6. Gradio UI

**Why**:
- simple syntax
- pre-built chat interface

### 7. Training Configuration

**Why**:
- Batch Size: 1 (with gradient accumulation) - because of limited memory
- Learning Rate: 2e-4 - standard learning rate
- Max Steps: 500 - due to time constraints
- Max Sequence Length: 512 - Handles QA pairs in this case

---

## Setup Instructions

### Required

- Python 3.8+
- Ollama installed and running

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd bose-QA-system
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Model Setup

1. **Train the model in Colab**:
   - Open `colab/DM8SE_SLM_Training.ipynb`
   - Run training cells for  fine-tuning
   - Download the trained model files

2. **Merge model weights**:
   - Open `colab/Merge_model.ipynb`
   - Merge LoRA adapters with base model
   - Save merged model

3. **Download model files**:
   - Ensure all model files are in `models/downloaded/`
   - Required files: `bose-slm.gguf`, `Modelfile`, tokenizer files

4. **Clone Ollama.cpp** (if converting from PyTorch):
   ```bash
   git clone https://github.com/ollama/ollama.git
   cd ollama
   ```

5. **Convert to GGUF format**:
   ```bash
   # Use llama.cpp conversion tools
   python convert.py <model_path> --outtype f16
   ```

6. **Create Modelfile**:
   - Modelfile is already provided in `models/downloaded/Modelfile`
   - Customize parameters if needed

7. **Import model into Ollama**:
   ```bash
   cd models/downloaded
   ollama create bose-slm -f Modelfile
   ```


### Running the UI

1. **Start Ollama** (if not running):
   ```bash
   ollama serve
   ```

2. **Launch Gradio interface**:
   ```bash
   python ui/app.py
   ```

3. **Access the interface**:
   - Open browser and  put `http://localhost:7860` in the address bar.

---

### Example Queries

- "What is the peak power handling of the speaker?"
- "What is the IP rating of the DM8SE loudspeaker?"
- "What is the Net Weight of a single DM8SE loudspeaker?"
- "If an installer bypasses the transformer, what is the Nominal Impedance of the speaker?"





