# Deforestation detection in Kinabatangan, Sabah

A geospatial machine learning pipeline that maps forest loss in the Kinabatangan district of Sabah, Malaysia, from Sentinel-2 satellite imagery. It compares a Random Forest baseline against a U-Net trained from scratch in PyTorch, and serves the U-Net predictions in an interactive Streamlit map.

## The problem

Sabah has lost a lot of lowland forest to oil palm. Kinabatangan is one of the most affected districts and is home to orangutans, proboscis monkeys and elephants. Tracking where clearing has happened normally needs manual interpretation of imagery, so this project tests how far an automated model can get using only free data.

## Data

| Source | Use |
|---|---|
| Sentinel-2 SR Harmonized (Google Earth Engine) | 10 m imagery, cloud-masked median composite for 2025 |
| Hansen Global Forest Change v1.11 | Labels: pixels with tree-cover loss in 2019 to 2023 |
| FAO GAUL level 2 | Kinabatangan district boundary |
| WDPA | Protected reserves, excluded from the study area |

Protected reserves cover about 33.7% of the district and account for under 3% of the Hansen loss, so they were masked out to keep the labels focused on unprotected land.

Each pixel has six input features: blue, green, red, near-infrared, NDVI and EVI. The area was exported in six tiles and cut into 256 x 256 patches, keeping patches where at least 30% of pixels had a valid label. That gave 1,091 patches. About 27% of labelled pixels are forest loss.

## Results

Patch-level 75/25 train/test split (random_state=42), so no patch appears in both sets.

| Model | Precision | Recall | Cohen's kappa |
|---|---|---|---|
| Random Forest (per-pixel) | 0.472 | 0.535 | 0.320 |
| U-Net | 0.501 | 0.776 | 0.395 |

The U-Net finds substantially more of the lost forest (recall 0.78 vs 0.54) at similar precision, mainly because it sees spatial context rather than single pixels.

## Method

- **Random Forest:** 200 trees, max depth 15, balanced class weights, trained on 200 sampled pixels per patch.
- **U-Net:** encoder 32/64/128, bottleneck 256, skip connections. Masked per-pixel BCE loss (pixels without a label are ignored), `pos_weight=2.7` for class imbalance, Adam at 1e-4, 15 epochs, batch size 8.
- **Inference:** windowed reads and streamed writes so the full region fits in Colab memory, then reprojection to EPSG:3857 for the web map.

## Repo structure

```
app.py                         Streamlit map viewer
requirements.txt
notebooks/
  01_gee_data_prep.ipynb       Earth Engine: composite, labels, export
  02_patches_random_forest.ipynb   Patch tiling and RF baseline
  03_unet_training_inference.ipynb U-Net, evaluation, inference, reprojection
```

## Running it

1. Run the notebooks in order in Google Colab. You need your own Earth Engine project ID (set it in notebook 01) and a Google Drive for intermediate files.
2. Notebook 03 writes `web_pred_*.tif` files. Download them into a `predictions_web/` folder next to `app.py`. They are large, so they are not in this repo.
3. Install and launch the app:

```
pip install -r requirements.txt
streamlit run app.py
```

The sidebar slider sets the minimum predicted probability that is drawn. Raising it shows only the pixels the model is most confident about.

## Limitations

- The model uses a single 2025 composite as input and predicts loss that happened between 2019 and 2023. It therefore learns what cleared land looks like, not the change between two dates. A proper change-detection setup would feed in before and after composites.
- Hansen flags any tree-cover loss, including logging, fire and plantation harvest. Calling it oil-palm clearing is an interpretation based on the region, not something the labels confirm.
- Hansen labels are themselves noisy, which caps the achievable scores.
- Precision around 0.5 means about half of flagged pixels are false alarms, so this is a screening aid and not a replacement for manual verification.
- Evaluation is a random patch split inside one region. Neighbouring patches can be spatially correlated, so performance on a different region would likely be lower.
