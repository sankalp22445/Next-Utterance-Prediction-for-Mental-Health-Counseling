# Next Utterance Prediction for Mental Health Counseling

**Authors:** Sankalp, Sarthak, Vivan

**Institution:** IIIT Delhi

**Contact:** sankalp22445@iiitd.ac.in, sarthak22453@iiitd.ac.in, vivan22581@iiitd.ac.in

## Project Overview

This project focuses on **Next Utterance Prediction (NUP)** in mental health counseling dialogues, where generating context-aware and emotionally aligned responses is critical. The system assists in therapeutic settings by predicting the therapist's next response based on conversation history.

We experiment with multilingual and instruction-tuned transformer models (`google/mt5-small`, `google/flan-t5-base`) and introduce a novel **Fusion Architecture** that combines transformer encoders with LSTM/GRU layers to capture conversational flow more effectively. Additionally, a sentiment classifier is integrated to steer generation toward emotionally consistent replies via post-processing.

---

## 📂 File Structure

The repository is organized as follows based on the experimental pipeline:

```text
Next-Utterance-Prediction-for-Mental-Health-Counseling/
├── 80_Report.pdf                  # Complete project report with methodology and results
├── README.md                      # Project documentation
├── baseline-1.ipynb               # Baseline experiments using T5-small
├── baseline-2.ipynb               # Secondary baseline experiments using Flan-T5-base
├── final-project(Lstm-gru).ipynb  # Implementation of the Fusion Architecture (Transformer + LSTM/GRU)
├── final-project.ipynb            # Main pipeline: LoRA fine-tuning, Curriculum Learning, and Back-Translation
└── sample_predictions.csv         # Generated outputs and evaluation metrics

```

---

## Methodology

### 1. Architectures

We explored two primary architectural setups:

* **Setup A (Efficient Fine-Tuning):** Uses `google/flan-t5-base` with **LoRA** (Low-Rank Adaptation) to inject trainable matrices into attention modules.
* **Setup B (Fusion Models):** Integrates transformer encoders with bidirectional **LSTM** and **GRU** layers. The RNN output is compressed and fed as input embeddings to the decoder to improve context tracking.

### 2. Training Strategies

* **Curriculum Learning:** Data is ranked by target text length and split into `Easy`, `Medium`, and `Hard` stages. The model trains sequentially on these stages.
* **Data Augmentation:** We employed back-translation (English  French  English) using MarianMT to create paraphrased inputs and increase dataset diversity.

### 3. Emotional Alignment (Post-Processing)

To ensure therapeutic safety:

1. **Sentiment Classification:** A DistilBERT model labels the sentiment of the generated response.
2. **Rewriting:** If the sentiment mismatches the target, a T5-based emotion rewriting model rephrases the response (e.g., "rephrase to be positive").

---

## Evaluation & Results

We evaluated models using BLEU, ROUGE, BERTScore, and BLEURT.

### Baseline Results

Initial benchmarks established using standard transformers:

| Model | BLEU | BERTScore F1 | BERTScore Precision |
| --- | --- | --- | --- |
| **T5-small** | 0.0020 | 0.8521 | 0.8659 |
| **Flan-T5-base** | 0.0033 | 0.8580 | 0.8706 |

### Advanced Model Comparison

Comparison of base models against Fusion architectures on the test set:

| Model | BLEU | ROUGE-L | BERTScore F1 | BLEURT |
| --- | --- | --- | --- | --- |
| **MT5 (Model1)** | 0.0012 | 0.0237 | 0.8193 | -1.3568 |
| **Flan-T5 (Model2)** | **0.0054** | **0.0642** | 0.8229 | **-1.3229** |
| **MT5 + LSTM/GRU** | 0.0028 | 0.0475 | **0.8250** | -1.3762 |
| **Flan-T5 + LSTM/GRU** | 0.0021 | 0.0453 | 0.8146 | -1.3860 |

**Findings:**

* **Semantic Quality:** The Fusion model (MT5 + LSTM/GRU) achieved the highest BERTScore in the advanced comparison, highlighting the value of combining linguistic structure with sequential memory.
* **Lexical Overlap:** Instruction-tuned T5 (Model2) demonstrated the strongest lexical overlap (BLEU/ROUGE).

---

## Installation and Usage

### Prerequisites

Ensure you have Python installed along with the following libraries (inferred from methodology):

* `torch`
* `transformers`
* `datasets`
* `peft`
* `scikit-learn`

### Running the Experiments

1. **Clone the repository:**
```bash
git clone https://github.com/sankalp22445/Next-Utterance-Prediction-for-Mental-Health-Counseling.git
cd Next-Utterance-Prediction-for-Mental-Health-Counseling

```


2. **Run Baselines:**
Open `baseline-1.ipynb` or `baseline-2.ipynb` to reproduce T5-small and Flan-T5-base results.
3. **Run Main Training (LoRA + Curriculum):**
Execute `final-project.ipynb` to train the instruction-tuned models using the curriculum learning stages.
4. **Run Fusion Models:**
Execute `final-project(Lstm-gru).ipynb` to train and evaluate the custom Transformer-RNN hybrid architectures.

---

## Future Work

* **Behavioral Codes:** Conditioning generation on predicted therapeutic codes (e.g., reflection, open questions).
* **Multi-Modal Inputs:** Integrating non-verbal cues (audio/video).
* **Human Evaluation:** Conducting expert evaluation to assess empathy and clinical appropriateness beyond automatic metrics.

---

## License & Citation

This project was conducted as research at IIIT Delhi. If you use this code or methodology, please cite:

> *Next Utterance Prediction for Mental Health Counseling*, Sankalp, Sarthak, Vivan, IIIT Delhi, 2025.
