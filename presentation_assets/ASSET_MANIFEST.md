# Presentation Asset Manifest

Bu klasordeki her dosya sunumda direkt kullanilsin diye slayt numarasiyla etiketlendi.

## Secilen Gercek Dataset Ornekleri

- Normal sample: `CR32a95e6-CR3dcad12-000058.avi` | EF=60.97 | Age=13 | Sex=F | Split=9
- Dysfunction sample: `CR32a95ef-CR32a97b8-000045.avi` | EF=21.95 | Age=0 | Sex=M | Split=7

## Slayt Bazli Asset Listesi

| Tag | Slayt | Kullanım | Kaynak |
|---|---|---|---|
| `S01_cover_echo_frame.png` | Slayt 1 | Kapak icin gercek echo frame | `data/pediatric_echo_avi/A4C/Videos/CR32a95e6-CR3dcad12-000058.avi` |
| `S01_cover_echo_loop_normal.gif` | Slayt 1 optional | Kapakta otomatik loop GIF | `data/pediatric_echo_avi/A4C/Videos/CR32a95e6-CR3dcad12-000058.avi` |
| `S02_ef_distribution.png` | Slayt 2 | EF histogram/dagilim | `results/figures/eda_ef_distribution.png` |
| `S03_architecture_pipeline.png` | Slayt 3 | 5 bloklu mimari diyagram | `generated` |
| `S04_real_ed_es_contour_grid.png` | Slayt 4 | Raw ED/ES + expert contour grid | `data/pediatric_echo_avi/A4C/Videos/CR32a95e6-CR3dcad12-000058.avi` |
| `S04_dae_reconstructions.png` | Slayt 4 | DAE original/noisy/reconstruction | `results/figures/dae_reconstructions.png` |
| `S04_normal_ED_contour.png` | Slayt 4 optional | ED contour tek gorsel | `data/pediatric_echo_avi/A4C/Videos/CR32a95e6-CR3dcad12-000058.avi` |
| `S04_normal_ES_contour.png` | Slayt 4 optional | ES contour tek gorsel | `data/pediatric_echo_avi/A4C/Videos/CR32a95e6-CR3dcad12-000058.avi` |
| `S06_ablation_barplot.png` | Slayt 6 | Ablation barplot | `results/figures/ablation_barplot.png` |
| `S06_normal_case_clip.gif` | Optional demo | Normal EF otomatik loop GIF, EF=61.0 | `data/pediatric_echo_avi/A4C/Videos/CR32a95e6-CR3dcad12-000058.avi` |
| `S06_normal_video_thumbnail.png` | Optional demo | Normal video thumbnail | `data/pediatric_echo_avi/A4C/Videos/CR32a95e6-CR3dcad12-000058.avi` |
| `S06_dysfunction_case_clip.gif` | Optional demo | Dysfunction otomatik loop GIF, EF=21.95 | `data/pediatric_echo_avi/A4C/Videos/CR32a95ef-CR32a97b8-000045.avi` |
| `S06_dysfunction_video_thumbnail.png` | Optional demo | Dysfunction video thumbnail | `data/pediatric_echo_avi/A4C/Videos/CR32a95ef-CR32a97b8-000045.avi` |
| `S07_scatter_pred_vs_true.png` | Slayt 7 | Prediction vs true scatter | `results/figures/scatter_pred_vs_true.png` |
| `S07_training_curves.png` | Slayt 7 | Training curves | `results/figures/training_curves.png` |
| `S08_temporal_evidence_heatmap.png` | Slayt 8 | Temporal evidence heatmap | `results/figures/attention_heatmap.png` |

## Kisa Kullanim

- Video istemezsen GIF yerine statik gerekiyorsa ayni slayttaki thumbnail/PNG dosyasini koy.
- Slayt 1: `S01_cover_echo_frame.png` veya `S01_cover_echo_loop_normal.gif`.
- Slayt 4: en guclu gorsel `S04_real_ed_es_contour_grid.png`.