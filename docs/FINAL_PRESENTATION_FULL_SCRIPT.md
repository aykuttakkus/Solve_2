# EchoCardioNet-Peds — Final Presentation Script

Bu dosya sunumun son halidir. Her slaytta:

- **Visual:** Canva/PPT'e koyulacak gorsel dosyasi
- **On-slide text:** Slaytta gorunecek kisa metin
- **Bottom sentence:** Slaytin altina koyulacak tek cumle
- **Speaker script:** Slayti anlatirken soylenebilecek metin

---

## Slide 1 — Title

**Title:**  
EchoCardioNet-Peds

**Subtitle:**  
Ejection Fraction Prediction from Pediatric Echocardiography Videos

**Visual:**  
`presentation_assets/S01_cover_echo_loop_normal.gif`

**On-slide text:**

- Stanford EchoNet-Pediatric dataset
- A4C echocardiography videos
- Deep learning regression model
- Output: EF percentage

**Bottom sentence:**  
**Input: cardiac video -> Output: EF (%)**

**Speaker script:**  
This project predicts ejection fraction directly from pediatric echocardiography videos. The input is a short A4C cardiac video, and the output is a continuous EF percentage. Therefore, this is not a classification task; it is a clinical regression problem.

---

## Slide 2 — Dataset Source

**Title:**  
Research Paper Dataset

**Visual:**  
`presentation_assets/S01_cover_echo_frame.png`  
Alternative: `results/figures/eda_sample_frames.png`

**On-slide text:**

- Dataset: **EchoNet-Pediatric**
- Source: **Stanford research paper, 2022**
- Not MNIST, not Kaggle baseline
- View used: **A4C**
- Task: **EF regression from video**

**Bottom sentence:**  
**We selected a real clinical research dataset instead of a simple benchmark.**

**Speaker script:**  
We selected EchoNet-Pediatric because it is a real clinical dataset from Stanford, published as a research paper. This is not a simple benchmark like MNIST and not a standard Kaggle baseline. We used the A4C view because it is clinically relevant for EF measurement.

---

## Slide 3 — Dataset Distribution

**Title:**  
Dataset Distribution

**Visual:**  
`presentation_assets/S02_ef_distribution_professional_histogram_clean.png`

**On-slide text:**

- Total A4C videos: **3,284**
- Train / Validation / Test: **2,580 / 336 / 368**
- EF mean +/- std: **60.9 +/- 10.5**
- Dysfunction group EF<40: **5.7% of train split**
- Strong clinical imbalance toward normal EF

**Bottom sentence:**  
**The dataset is clinically imbalanced, so low-EF detection is a meaningful challenge.**

**Speaker script:**  
The dataset contains 3,284 A4C videos. The train, validation, and test split is 2,580, 336, and 368 videos. The important point is the imbalance: only 5.7 percent of the train split belongs to the EF below 40 dysfunction group. This makes low-EF detection clinically meaningful and more difficult than a balanced toy dataset.

---

## Slide 4 — 5-Block Pipeline

**Title:**  
5-Block Deep Learning Pipeline

**Visual:**  
Use the pipeline table/diagram created in Canva, or:  
`presentation_assets/S03_architecture_pipeline.png`

**On-slide text:**

- **Conv DAE:** denoised frame representation
- **CNN Encoder:** spatial cardiac features
- **BiLSTM:** long-range temporal context
- **GRU:** video-level sequence summary
- **Sparse AE:** regularized bottleneck

**Shape flow:**

- Input: `(B, T, 1, 112, 112)`
- DAE + CNN: `(B, T, 512)`
- BiLSTM: `(B, T, 512)`
- GRU: `(B, 256)`
- Sparse AE: `(B, 128)`
- Output: `EF (%)`

**Bottom sentence:**  
**The model combines CNN, recurrent layers, and autoencoders in one end-to-end video regression pipeline.**

**Speaker script:**  
The model has five main blocks. The Conv DAE and CNN work at the frame level. Their outputs are concatenated into a 512-dimensional sequence representation. Then the BiLSTM models long-range temporal context, the GRU summarizes the video, and the Sparse Autoencoder creates a regularized bottleneck before EF prediction.

---

## Slide 5 — Required Algorithm 1: CNN Block

**Title:**  
Required Algorithm 1: CNN Block

**Subtitle:**  
CNN for Spatial Feature Extraction

**Visual:**  
`presentation_assets/S_req1_cnn_performance_only.png`

**On-slide text:**

- Satisfies the **CNN** requirement.
- Processes each echo frame independently.
- Extracts spatial cardiac anatomy features.
- Learns ventricle shape, wall boundaries, and chamber structure.
- CNN-based baseline: **MAE 6.815**, **RMSE 12.078**, **R2 -0.087**

**Bottom sentence:**  
**CNN provides the spatial feature backbone of the model.**

**Speaker script:**  
This block satisfies the CNN requirement. The CNN processes each echo frame independently and extracts spatial cardiac anatomy features such as ventricular shape, wall boundaries, and chamber structure. In our ablation setup, the CNN-based baseline starts with MAE 6.815, RMSE 12.078, and R2 -0.087.

---

## Slide 6 — Required Algorithm 2: RNN / LSTM / GRU Block

**Title:**  
Required Algorithm 2: RNN / LSTM / GRU Block

**Subtitle:**  
BiLSTM + GRU for Temporal Modeling

**Visual:**  
`presentation_assets/S_req2_rnn_performance_only.png`

**On-slide text:**

- Satisfies the **RNN / LSTM / GRU** requirement.
- **BiLSTM** reads the video forward and backward.
- **GRU** summarizes the sequence into one video vector.
- RMSE improved from **12.066 -> 11.880** after adding GRU.
- Purpose: learn cardiac motion across frames.

**Bottom sentence:**  
**Recurrent blocks model the temporal structure of the echo video.**

**Speaker script:**  
This block satisfies the RNN, LSTM, and GRU requirement. We use BiLSTM to read the video in both temporal directions, which helps capture long-range cardiac motion. Then the GRU summarizes the sequence into a compact video-level representation. In the ablation study, adding GRU reduced RMSE from 12.066 to 11.880.

---

## Slide 7 — Required Algorithm 3: Autoencoder Block

**Title:**  
Required Algorithm 3: Autoencoder Block

**Subtitle:**  
Conv DAE + Sparse AE for Robust Representation

**Visual:**  
`presentation_assets/S_req3_autoencoder_performance_only.png`

**On-slide text:**

- Satisfies the **Autoencoder** requirement.
- **Conv DAE** learns denoised frame representations.
- **Sparse AE** creates a regularized bottleneck.
- Reduces sensitivity to noisy ultrasound frames.
- Sparse AE reduced RMSE from **11.852 -> 11.047**.

**Bottom sentence:**  
**Autoencoders improve representation quality and regularization.**

**Speaker script:**  
This block satisfies the Autoencoder requirement. We use two autoencoder-style components. The Conv DAE learns more robust frame representations from noisy ultrasound images. The Sparse AE creates a regularized bottleneck before regression. In the ablation study, Sparse AE reduced RMSE from 11.852 to 11.047.

---

## Slide 8 — Training Loss

**Title:**  
Training Loss During Fine-Tuning

**Visual:**  
`presentation_assets/S07_training_loss_only.png`

**On-slide text:**

- Full model fine-tuned for **15 epochs**
- Training loss: **231.75 -> 28.55**
- Loss reduction: **87.7%**
- Best validation MAE: **5.83**
- DAE was pre-trained before fine-tuning

**Small table:**

| Metric | Start | Final / Best |
|---|---:|---:|
| Training Loss | 231.75 | 28.55 |
| Validation MAE | 6.58 | 5.83 |

**Bottom sentence:**  
**Training loss decreased strongly during fine-tuning.**

**Speaker script:**  
This graph shows the fine-tuning process. Training loss decreased from 231.75 in the first epoch to 28.55 after 15 epochs, which is an 87.7 percent reduction. The best validation MAE reached 5.83. This shows that the full model received a strong optimization signal during training.

---

## Slide 9 — Ablation Study

**Title:**  
Ablation Study

**Visual:**  
`presentation_assets/S06_ablation_academic_figure.png`

**Small table:**

| Metric | Baseline | Final | Change |
|---|---:|---:|---:|
| MAE | 6.815 | 6.742 | down 0.073 |
| RMSE | 12.078 | 9.969 | down 2.109 |
| R2 | -0.087 | 0.260 | up 0.347 |

**On-slide text:**

- Baseline model: CNN + UniLSTM
- Full model: DAE + CNN + BiLSTM + GRU + Sparse AE
- RMSE reduced by **17.5%**
- R2 improved from negative to positive

**Bottom sentence:**  
**Ablation shows that the full model gives the strongest overall result.**

**Speaker script:**  
This ablation study compares the baseline and block-wise variants. The baseline uses a simpler CNN and UniLSTM setup. The full model includes DAE, CNN, BiLSTM, GRU, and Sparse AE. The key result is that RMSE decreased from 12.078 to 9.969, which is a 17.5 percent reduction. R2 also improved from negative to positive.

---

## Slide 10 — Final Test Results

**Title:**  
Final Test Results

**Visual:**  
Use one of these:

- `presentation_assets/S08_final_test_results_summary.png`
- `presentation_assets/S06_baseline_vs_final_single_figure_clean.png`
- `presentation_assets/S07_scatter_pred_vs_true.png`

**Best model:**  
**Full EchoCardioNet-Peds — 15-epoch fine-tuned model**

**Result table:**

| Metric | Test Result |
|---|---:|
| MAE | 6.742 |
| RMSE | 9.969 |
| R2 | 0.260 |
| AUROC EF<40 | 0.824 |

**On-slide text:**

- Best result came from the **full 5-block model**.
- Test MAE: **6.742 EF points**
- Test RMSE: **9.969 EF points**
- EF<40 detection AUROC: **0.824**
- Final model reduced RMSE by **17.5%** compared with baseline.

**Bottom sentence:**  
**The full model achieved the best overall test performance.**

**Speaker script:**  
The best result came from the full EchoCardioNet-Peds model, fine-tuned for 15 epochs. On the test set, it achieved MAE 6.742, RMSE 9.969, R2 0.260, and AUROC 0.824 for EF below 40 detection. This is our final result, and it shows that the complete five-block model performs best overall.

**Closing:**  
Thank you. I am ready for your questions.
