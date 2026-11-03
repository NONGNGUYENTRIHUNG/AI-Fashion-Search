# Fashion Lens · AI Fashion Search

**A local AI application for clothing recognition and image-and-text search.**

Fashion Lens lets users upload a clothing photo, enter a description, or combine both to find similar items in a local image catalog. It also uses a custom supervised classifier to recognize six clothing categories.

Built as an end-to-end Data Science project, it connects dataset preparation, pretrained image representations, supervised learning, evaluation, and model inference to a React website through a FastAPI backend.

> **Repository status:** This repository currently documents the locally implemented application. Application source code and a home-screen screenshot have not yet been uploaded. Setup commands below apply once the project source is present.

## Purpose

Searching by keywords alone can be difficult when someone has a clothing photo but does not know how to describe it. Fashion Lens explores how visual and text representations can help retrieve similar clothing, while a separately trained classifier adds category recognition.

For example, a user can:
- Search for **“a blue shirt”** to rank catalog images by relevance.
- Upload a photo of shoes to find visually similar catalog items.
- Upload a clothing photo and see predictions from the trained category classifier.
- Save interesting results in a browser-local collection.

This is a portfolio and research prototype. It searches the catalog supplied to the application; it does not search online stores, check prices, generate clothing images, or recommend complete outfits.

## Features

| Feature | What it does |
| --- | --- |
| Photo search | Encodes an uploaded image and retrieves visually similar catalog items. |
| Text search | Matches a description against catalog image representations. |
| Combined search | Averages normalized image and text query vectors to retrieve matching items. |
| Clothing classification | Predicts dress, jeans, shirt, shoes, shorts, or T-shirt using a trained classifier. |
| Saved collection | Saves favorite catalog items in the current browser using local storage. |
| Dataset preparation | Selects labeled images, removes exact pixel duplicates, and creates separate data splits. |
| Evaluation | Compares regularization settings on validation data and reports performance on held-out test images. |
| Local operation | Runs without a paid AI API; installation and the initial model/dataset downloads require internet access. |

Search returns up to 12 ranked items by default. Classification is shown for photo queries when the trained model and a nonempty catalog are available. Search and classification have different jobs: training the classifier does not automatically add images to the catalog or improve retrieval ranking.

## Technology stack

| Technology | Role |
| --- | --- |
| **React** | Interactive website, uploads, search results, and saved favorites. |
| **Vite** | Frontend development server, API proxy, and production build. |
| **FastAPI + Uvicorn** | Python API, request validation, and local backend server. |
| **CLIP ViT-B/32** | Pretrained image and text encoder for a shared embedding space. |
| **Sentence Transformers** | Loads CLIP and produces normalized embeddings. |
| **PyTorch** | Runs the neural encoder, using CUDA when available. |
| **scikit-learn** | Trains and evaluates the logistic regression clothing classifier. |
| **Hugging Face Datasets** | Streams labeled product images for dataset preparation. |
| **NumPy** | Vector operations and cosine-similarity ranking. |
| **Pillow** | Decodes, validates, and prepares image files. |
| **Joblib** | Saves and loads the trained classifier artifact. |
| **pytest + HTTPX/TestClient** | Checks API validation and request handling. |

## How the AI works

### 1. Clothing retrieval

The backend uses `sentence-transformers/clip-ViT-B-32` to encode catalog photos and user queries into a shared vector space.

1. Catalog images are embedded and normalized.
2. A text description, uploaded photo, or both are embedded.
3. For combined queries, image and text vectors are averaged and normalized again.
4. Cosine similarity ranks the catalog items.
5. The website displays the highest-ranking matches.

Catalog embeddings are cached in memory and refreshed when catalog files change or the backend restarts. Similarity scores are ranking values, **not accuracy percentages or calibrated confidence scores**. Combined search blends representations; it is not an image-editing or attribute-replacement model.

### 2. Supervised clothing classification

A separate logistic regression model is trained on frozen CLIP image embeddings.

- **Inputs:** CLIP features extracted from labeled clothing photos.
- **Targets:** Six clothing categories.
- **Training:** Class-balanced logistic regression with up to 2,000 optimization iterations.
- **Model selection:** Compare `C = 0.1, 1.0, 10.0` using validation macro-F1.
- **Selected setting:** `C = 10.0`.
- **Inference:** Encode a new photo and pass its features to the saved classifier.

**CLIP itself is not fine-tuned.** The supervised component learned in this project is the clothing classifier built on top of its representations.

## Dataset and preparation

Dataset: [Fashion Product Images Small on Hugging Face](https://huggingface.co/datasets/ashraq/fashion-product-images-small).

The preparation script targets up to 500 usable images per category. The completed experiment used **2,939 images**:

| Split | Images | Use |
| --- | ---: | --- |
| Training | 2,089 | Fit the classifier. |
| Validation | 422 | Select regularization strength. |
| Test | 428 | Evaluate the selected approach on held-out images. |

The six categories are **dress, jeans, shirt, shoes, shorts, and tshirt**. Casual, formal, and sports shoes are combined into the shoes class.

Preparation removes exact pixel duplicates globally. Within each category, images with the same normalized product display name stay together in one split. Approximately 70% of product-name groups go to training, 15% to validation, and 15% to testing; image counts vary with group sizes and available examples.

The script records image assignments and preparation details in `manifest.json` and `dataset_report.json`. Sampling uses a seeded streaming shuffle buffer rather than a uniform shuffle of the full dataset.

## Model results

The following results come from the completed local training and evaluation run:

| Metric | Validation | Test |
| --- | ---: | ---: |
| Accuracy | 98.58% | **99.07%** |
| Macro-F1 | 98.61% | **99.09%** |
| Images classified correctly | 416 / 422 | **424 / 428** |

The validation majority-class baseline accuracy was **16.59%**.

### Test results by category

| Category | Precision | Recall | F1 | Test images |
| --- | ---: | ---: | ---: | ---: |
| Dress | 98.63% | 100.00% | 99.31% | 72 |
| Jeans | 100.00% | 100.00% | 100.00% | 58 |
| Shirt | 98.61% | 98.61% | 98.61% | 72 |
| Shoes | 100.00% | 98.75% | 99.37% | 80 |
| Shorts | 98.67% | 100.00% | 99.33% | 74 |
| T-shirt | 98.59% | 97.22% | 97.90% | 72 |

**These results measure six-category classification on held-out images from this product-photo dataset.** They do not measure retrieval relevance, outfit quality, or accuracy on arbitrary real-world photos. The reported split also cannot establish whether individual images were absent from CLIP's original pretraining data.

## Run locally

Once application source is uploaded, use the commands below.

### Requirements

- Python **3.11 or 3.12**
- Node.js and npm
- Internet for initial dependency/model downloads
- Optional compatible NVIDIA GPU for accelerated inference

The commands below use **Windows PowerShell** and a virtual environment inside the project root. An existing environment outside the project can also be used by adjusting the Python executable path.

### Backend — terminal 1

From the directory containing `requirements.txt` and the `backend` folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Use `py -3.11` if Python 3.11 is installed instead.

### Frontend — terminal 2

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

Open the Local URL printed by Vite, usually **http://localhost:5173**. Keep both terminals running.

- Backend API documentation: **http://127.0.0.1:8000/docs**
- Health check: **http://127.0.0.1:8000/api/health**

### Add a searchable catalog

Copy JPG, PNG, or WebP images into `data/catalog/`, directly inside that folder. Use unique, descriptive filenames. Refresh the website and search.

The first search downloads CLIP if needed and embeds the catalog. Larger catalogs take longer to index. Uploaded query photos are processed in memory rather than saved by the application. Catalog files remain local, and favorite identifiers are stored in browser local storage.

## Train the classifier

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m ml.prepare_fashion --per-class 500
.\.venv\Scripts\python.exe -u -m ml.train --data data/fashion_v1
```

The dataset revision must contain the Parquet data files. The local experiment used the dataset's `main` revision after an earlier empty revision was corrected; do not use the obsolete revision from the initial add-on. For future reproducible runs, record the resolved dataset commit.

After selecting model settings using validation data, run the final test:

```powershell
.\.venv\Scripts\python.exe -u -m ml.train --data data/fashion_v1 --test
```

The current trainer repeats training and validation selection before optional test evaluation. Keep model choices fixed when evaluating test performance.

Outputs:
- `models/classifier.joblib` — trained classifier and encoder identifier.
- `models/metrics.json` — validation trials and optional test report.

Training replaces those artifacts; preserve earlier runs before starting another experiment. Restart the backend and upload a photo to use the trained classifier in the website. Only load trusted, locally trained Joblib artifacts.

## Project structure

Expected application layout:

```text
backend/
  main.py              # API endpoints and image validation
  engine.py            # CLIP loading, retrieval, and classification
frontend/
  src/main.jsx         # React interface and favorites
  src/style.css        # Responsive styling
ml/
  prepare_fashion.py   # Dataset selection, deduplication, and splits
  train.py             # Supervised training and evaluation
tests/
  test_api.py          # API checks
data/
  catalog/             # Searchable images
  fashion_v1/          # Prepared train/val/test data and reports
models/                # Local classifier artifact and metrics
requirements.txt
requirements-dev.txt
requirements-training.txt
```

## Checks

The initial implementation passed **seven API tests** and a **Vite production build**. API tests cover health, missing input, invalid images, oversized uploads, image/text request integration, an empty catalog, and invalid result limits. Model inference is mocked in the request integration test; this does not prove model accuracy.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm.cmd run build
```

## Current limitations

- The training dataset contains low-resolution product photos, so those images look blurry when enlarged. The app can display higher-resolution catalog originals when supplied.
- Product-image performance does not guarantee performance on cluttered backgrounds, multiple garments, or phone photos.
- The classifier always predicts among six known categories; it has no validated unknown-object rejection mechanism, and its probabilities are not calibrated.
- Product-name grouping and exact deduplication reduce leakage but do not eliminate near-duplicates or related products with different names.
- Retrieval uses pretrained CLIP; its quality has not been measured with labeled relevance judgments or Recall@K.
- Favorites do not currently train a personalized recommender.
- Catalog indexing is in-memory; database-backed storage, persistent vector indexes, accounts, and public deployment are not implemented.
- No outfit-compatibility model, virtual try-on, background segmentation, live shopping integration, or wardrobe planner is implemented.

## Future directions

- Evaluate category predictions on independently collected camera photos.
- Build a small labeled retrieval benchmark and report Recall@K.
- Use high-resolution, appropriately licensed catalog images.
- Add persistent wardrobe management and user-corrected labels.
- Investigate segmentation and encoder fine-tuning when error analysis supports them.
- Train outfit compatibility or personalized ranking only with suitable labels and interaction data.

These are proposed improvements, not current features.

## Data and credits

- [CLIP](https://github.com/openai/CLIP)
- [Sentence Transformers image search](https://sbert.net/examples/sentence_transformer/applications/image-search/)
- [Fashion Product Images Small](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
- [Original fashion product dataset project](https://github.com/paramaggarwal/fashion-product-images-dataset)

Dataset images and pretrained weights are separate from this project's code. Review their respective terms before redistribution or commercial use.

**Author:** [Nguyen Tri Hung Nong](https://github.com/NONGNGUYENTRIHUNG)
