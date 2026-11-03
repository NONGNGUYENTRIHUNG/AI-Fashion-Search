# Fashion Lens: first supervised training experiment

Run these commands from the repository root. The trained classifier is already included; these steps reproduce training with a newly prepared dataset.

## Goal and honest scope
Train a six-category classifier: shirt, tshirt, jeans, dress, shoes, shorts. Up to 500 unique images per class (up to 3,000 total); categories with fewer usable images use all available examples. The program prints exact counts. About 70% of product-name groups go to training, 15% validation, 15% test. Counts can differ because groups have different sizes.

This trains logistic regression using frozen pretrained CLIP image features. It does not fine-tune CLIP and does not change the search ranking. The website displays your trained category predictions when you upload a photo and your catalog has images.

## 1. Install the dataset dependency
Keep the existing website/backend terminals open. Open a third PowerShell terminal:

```powershell
cd "C:\path\to\AI-Fashion-Search"
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

## 2. Download and split
```powershell
.\.venv\Scripts\python.exe -m ml.prepare_fashion --per-class 500
```
Wait for READY. Output: data/fashion_v1, including manifest.json and dataset_report.json. Leave these together for reproducibility. A .partial directory indicates an unfinished download, not usable training data. Existing output folders are never overwritten. For a new experiment choose another --output path and pass that path to training.

## 3. Train and validate
```powershell
.\.venv\Scripts\python.exe -u -m ml.train --data data/fashion_v1
```
Image embedding can take a while on CPU; the original trainer prints metrics at the end. Keep the terminal open. It compares three regularization settings on validation macro-F1 and saves models/classifier.joblib plus models/metrics.json. It replaces any earlier classifier/metrics, so back those up before another experiment. It does not train on test images.

Send the validation report to your project lead before running final testing. Compare macro-F1, per-class recall and majority baseline accuracy. We have not measured this dataset's accuracy yet.

## 4. Final test (after model choices are fixed)
```powershell
.\.venv\Scripts\python.exe -u -m ml.train --data data/fashion_v1 --test
```
The current trainer repeats deterministic training and validation selection, then evaluates test data once. Do not repeatedly adjust models based on test scores. This command also replaces classifier.joblib and metrics.json.

## 5. Try your trained model in the website
Restart the backend with the same command you used previously. Keep your catalog photos. Upload a clothing photo, click Find my style, and scroll to the category predictions above the results. Text-only search has no classification output. The classifier always chooses among six known categories, even for an unrelated image; scores are not calibrated confidence guarantees.

## Dataset and limitations
Source: https://huggingface.co/datasets/ashraq/fashion-product-images-small
Dataset revision: main (record the resolved commit for future reproducibility)
This is a public mirror with approximately 44,000 low-resolution product images and articleType labels. Original project: https://github.com/paramaggarwal/fashion-product-images-dataset . The mirror has limited license documentation; check original dataset terms before redistributing images or using commercially. This add-on contains only code, not dataset images.
Exact pixel duplicates are removed, and identical normalized product names within a category stay in one split. This reduces leakage, but related garments and near-duplicates with different names may remain. The subset is sampled with a seeded streaming shuffle buffer, not a uniform sample of the complete dataset. Small product photos do not represent every real-world clothing photo. After evaluation, inspect mistakes and test separately on your own images.

## Verification
The preparation/splitting logic was tested on synthetic rows: class preservation, no group overlap, global pixel deduplication, no output overwrite. The project author completed dataset preparation and model training locally. The resulting test accuracy is 99.07% on 428 product images; the report is in models/metrics.json. This publication check reran seven API tests and the React build, but did not independently rerun model training.
