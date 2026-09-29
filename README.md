# 📱 SMS Spam Filter

Classifies SMS messages as **spam** or **ham** (not spam), comparing three approaches of increasing complexity on the same dataset — a classical ML baseline, a word-embedding + LSTM model, and a fine-tuned BERT transformer.

<!-- 🔗 Live demo: [add once deployed — see "Future Improvements" below] -->

## 📌 Overview

Spam-filter projects usually stop at one Naive Bayes model. This one goes further by implementing three fundamentally different approaches on the same dataset and comparing how much accuracy actually improves as model complexity increases — from a bag-of-words baseline, to embeddings + a recurrent network, to a fully fine-tuned transformer.

## 🗂️ Dataset

- **Source:** Kaggle `sms-spam-dataset`, loaded via `kagglehub`
- **Size:** ~5.5k messages, roughly 87% ham / 13% spam (imbalanced)
- **Key columns used:**
  - `Message` — raw SMS text
  - `spamORham` — original label (`spam` / `ham`)
  - `message_clean` — cleaned, stopword-removed, stemmed text
  - `target_encoded` — label encoded to 0/1

## 🧠 Approaches

| Model | Text input | Idea |
|-------|-----------|------|
| Multinomial Naive Bayes | `message_clean` (CountVectorizer / TF-IDF) | Fast, classic spam-filter baseline |
| Bi-LSTM + GloVe | `message_clean` (tokenized, padded sequences) | Pretrained word embeddings feeding a recurrent network |
| BERT (`bert-base-uncased`) | `Message` (raw text) | Fine-tuned transformer — higher expected accuracy, higher compute cost |

**Why BERT uses raw text instead of `message_clean`:** BERT does its own subword tokenization and is trained on natural language, so stemming and stopword removal — which helps a bag-of-words model — actually throws away information it relies on.

## 📓 Notebook Structure

1. **EDA** — class balance, message length distributions, word clouds
2. **Text cleaning** — regex cleanup, stopword removal, stemming (`message_clean`)
3. **Naive Bayes baseline** — CountVectorizer/TF-IDF + `MultinomialNB`
4. **GloVe + Bidirectional LSTM** — Keras `Sequential` model with a pretrained embedding matrix
5. **BERT fine-tuning** — `AutoTokenizer` / `TFAutoModelForSequenceClassification` (`bert-base-uncased`), fine-tuned end-to-end

## ⚙️ Requirements

```
pandas
numpy
matplotlib
seaborn
plotly
wordcloud
nltk
spacy
scikit-learn
tensorflow
keras
tf-keras
transformers==4.49.0
kagglehub
tqdm
```

Most are pre-installed on Google Colab; the notebook also runs its own `!pip install` cells for the rest.

## 🚀 Running It

- Built for **Google Colab**.
- Data loading uses `kagglehub`, which will prompt for Kaggle credentials if not already set up.
- Naive Bayes and GloVe/LSTM sections run fine on the default CPU runtime.
- **Before running the BERT section**, switch the runtime to GPU: `Runtime > Change runtime type > T4 GPU`. The notebook's accelerator defaults to TPU, but HuggingFace's TF model classes fine-tune far more reliably on GPU than on TPU without a `TPUStrategy` wrapper.
- After switching runtimes, restart the kernel and run all cells top to bottom.

### Keras 2 vs. Keras 3 compatibility note

Modern TensorFlow's `tf.keras` points to Keras 3, but `transformers`' TF model classes (`TFAutoModelForSequenceClassification`) are still built on the legacy Keras 2 API, packaged separately as `tf_keras`. Compiling the BERT model with a `tf.keras` optimizer raises `Could not interpret optimizer identifier`. The BERT section works around this by building its optimizer and loss from `tf_keras` instead — this only affects that section; the GloVe/LSTM model still uses plain `keras`/`tf.keras`.

## 📤 Outputs

- The fine-tuned BERT model and tokenizer are saved to `bert_spam_model/` via `save_pretrained()`.
- Each model's evaluation (`classification_report`, confusion matrix, and — for LSTM/BERT — learning curves) is printed/plotted in its own section of the notebook.

## ⚠️ Things to Keep in Mind

- The dataset is imbalanced (~87% ham / 13% spam). `train_test_split(..., stratify=...)` preserves that ratio between splits; if a model biases toward predicting "ham", `class_weight` in `.fit()` is the usual fix.
- BERT fine-tuning uses a small learning rate (`2e-5`) and few epochs (`3`), since it converges quickly and overfits fast on a dataset this size (~5.5k messages).

## 🔮 Future Improvements

- [ ] Deploy as a live demo (Streamlit/Gradio on Hugging Face Spaces) so recruiters can try it directly
- [ ] Add a `/predict` API endpoint (FastAPI)
- [ ] Add the results table above with real numbers
- [ ] Address class imbalance more rigorously (e.g. SMOTE) and compare against the `class_weight` approac
---

*Part of my BCA Hons in AI portfolio.*
