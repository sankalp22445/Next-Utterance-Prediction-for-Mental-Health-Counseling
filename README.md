# Next Utterance Prediction for Mental Health Counseling

**Authors:** Sankalp, Sarthak, Vivan (IIIT Delhi)

## Project Overview

This project focuses on **Next Utterance Prediction (NUP)** in mental health counseling dialogues. The goal is to develop an AI system capable of generating context-aware, empathetic, and emotionally aligned responses to assist in therapeutic settings.

Building upon standard Transformer models, this research introduces a **Fusion Architecture** that combines Transformer encoders with recurrent layers (LSTM/GRU) to better capture conversational flow. We also implement a sentiment-aware post-processing pipeline to ensure the emotional consistency required for mental health support.

### Key Features

* **Models:** Comparison of T5-small, google/mt5-small, and instruction-tuned google/flan-t5-base.
* **Architecture:** Novel Fusion Model integrating Transformer encoders with Bi-LSTM/GRU layers.
* **Optimization:** Utilizes **LoRA** (Low-Rank Adaptation) for efficient fine-tuning.
* **Data Strategy:** Implements **Back-Translation** (En  Fr  En) for augmentation and **Curriculum Learning** based on response difficulty.
* **Safety:** A sentiment-classifier-guided rewriting step to enforce emotional alignment.

---

## Methodology & Architecture

### 1. The Fusion Architecture

To enhance the sequential modeling capabilities of standard Transformers, we developed a custom architecture.

* **Backbone:** T5 or mT5 encoder.
* **Memory Integration:** The encoder output is passed through **Bidirectional LSTM/GRU layers** to capture long-term context and conversational flow.
* **Decoder:** The compressed output from the fusion layers serves as input embeddings for the Transformer decoder.

### 2. Training Pipeline

* **Data Augmentation:** Paraphrasing inputs using MarianMT back-translation to increase semantic variance.
* **Curriculum Learning:** The dataset is split into `Easy`, `Medium`, and `Hard` stages based on target text length. The model trains sequentially on these stages to stabilize convergence.
* **Instruction Tuning:** For Flan-T5, inputs are prepended with prompts (e.g., "Respond appropriately:").

### 3. Inference & Post-Processing

* **Decoding:** Beam search with repetition penalties.
* **Sentiment Alignment:**
1. **Detection:** A DistilBERT classifier analyzes the sentiment of the generated response vs. the ground truth.
2. **Rewriting:** If a mismatch occurs, a T5-based emotion rewriting model rephrases the output to match the target sentiment (e.g., "rephrase to be positive").



---

## Experimental Results

We evaluated models using **BLEU**, **ROUGE**, **BERTScore**, and **BLEURT**.

### Baseline Performance

| Model | BLEU | BERTScore F1 | BERTScore Precision |
| --- | --- | --- | --- |
| **T5-small** | 0.0020 | 0.8521 | 0.8659 |
| **Flan-T5-base** | 0.0033 | 0.8580 | 0.8706 |

### Advanced Model Comparison (Test Set)

*Models below utilized the custom training loop (Setup B).*

| Model Configuration | BLEU | ROUGE-L | BERTScore F1 | BLEURT |
| --- | --- | --- | --- | --- |
| **MT5 (Base)** | 0.0012 | 0.0237 | 0.8193 | -1.3568 |
| **Flan-T5 (Base)** | **0.0054** | **0.0642** | 0.8229 | **-1.3229** |
| **MT5 + LSTM/GRU Fusion** | 0.0028 | 0.0475 | **0.8250** | -1.3762 |
| **Flan-T5 + LSTM/GRU Fusion** | 0.0021 | 0.0453 | 0.8146 | -1.3860 |

**Key Observations:**

* **Semantic vs. Lexical:** All models achieved high semantic similarity (BERTScore ) but low lexical overlap (BLEU), indicating the models often paraphrase correctly rather than matching words exactly.
* **Fusion Impact:** The LSTM/GRU fusion improved semantic capture for the **MT5** model but did not benefit the already instruction-tuned Flan-T5.
* **Best Overall:** **Flan-T5** demonstrated the strongest balance of lexical overlap and learned quality (BLEURT).

---

## Installation & Usage

### Prerequisites

* Python 3.8+
* PyTorch
* Hugging Face Transformers & Datasets
* PEFT (for LoRA)

### Installation

```bash
git clone https://github.com/sankalp22445/Next-Utterance-Prediction-for-Mental-Health-Counseling.git
cd Next-Utterance-Prediction-for-Mental-Health-Counseling
pip install -r requirements.txt

```

### Training

To train the model using the curriculum learning strategy:

```bash
# Example command - adjust arguments as per your codebase
python train.py --model_name "google/flan-t5-base" --use_lora True --curriculum True

```

### Inference

To generate a response for a specific dialogue context:

```bash
python generate.py --input_text "Therapist: How are you? [SEP] Client: I feel overwhelmed."

```

---

## Dataset

The project utilizes structured therapy dialogues containing `input_text` (history) and `target_text` (next utterance).

* **Input:** "Therapist: Hi... [SEP] Client: ..."
* **Target:** "Tell me more about..."

*Note: The specific dataset used in this research was provided internally. Please refer to `data/README.md` for information on public equivalents or access.*

---

## Future Work

* **Behavioral Codes:** Conditioning generation on therapeutic strategies (e.g., Reflection, Open Question).
* **Multimodal Integration:** Incorporating audio prosody or video cues.
* **Human Evaluation:** Validating empathy and safety with clinical experts.

---

## Contributors

* **Sankalp** (sankalp22445@iiitd.ac.in)
* **Sarthak** (sarthak22453@iiitd.ac.in)
* **Vivan** (vivan22581@iiitd.ac.in)

---

### Citation

If you find this repository useful, please cite our project report:

> *Next Utterance Prediction for Mental Health Counseling*, IIIT Delhi, 2025.
