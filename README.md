<p align="center">
  <h1 align="center">◉ Antriksha Search Engine (ASE)</h1>
  <p align="center">
    <strong>Cross-Modal Satellite Image Retrieval powered by RemoteCLIP &amp; FAISS</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#demo">Demo</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#dataset">Dataset</a> •
    <a href="#usage">Usage</a> •
    <a href="#project-structure">Project Structure</a> •
    <a href="#tech-stack">Tech Stack</a> •
    <a href="#author">Author</a>
  </p>
</p>

---

## Overview

**Antriksha Search Engine (ASE)** is a cross-modal satellite image retrieval system that lets you search a gallery of **16,000+ SAR and Optical satellite images** using either **natural-language text** or an **uploaded image**.

It uses **RemoteCLIP** (ViT-B/32) — a domain-specific vision-language model fine-tuned for remote sensing — to encode queries and gallery images into a shared 512-dimensional embedding space. At query time, **FAISS** performs sub-millisecond nearest-neighbor lookups across the indexed gallery, and a **bifurcated routing layer** ensures true cross-modal retrieval (SAR → Optical or Optical → SAR).

The entire application is served through a polished, dark-themed **Streamlit** dashboard.

---

## Features

| Capability | Description |
|---|---|
| 🔤 **Text → Image** | Describe a scene in natural language and retrieve the most relevant satellite images |
| 🖼️ **Image → Image** | Upload a SAR or Optical image and find cross-modal or same-modal matches |
| ⚡ **Real-Time Search** | Sub-100 ms retrieval latency using FAISS IndexFlatL2 (exact nearest neighbor) |
| 🛰️ **Bifurcated Routing** | Separate FAISS indexes for SAR and Optical modalities to eliminate modality collapse |
| 📊 **CSV Export** | Download search results as a structured mission report |
| 🎨 **Premium Dark UI** | Linear/Vercel-inspired dark theme with Inter + JetBrains Mono typography |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                   (Streamlit — app.py)                        │
│                                                              │
│   ┌─────────────────┐        ┌─────────────────────────┐     │
│   │  Text Query      │        │  Image Upload            │    │
│   └────────┬─────────┘        └────────────┬─────────────┘    │
│            │                               │                  │
│            ▼                               ▼                  │
│   ┌────────────────────────────────────────────────────┐      │
│   │         RemoteCLIP ViT-B/32 Encoder                │      │
│   │          (extractor.py — 512-D vectors)             │      │
│   └────────────────────┬───────────────────────────────┘      │
│                        │                                      │
│            ┌───────────┴──────────┐                           │
│            ▼                      ▼                           │
│   ┌─────────────────┐   ┌─────────────────┐                  │
│   │  FAISS Index     │   │  FAISS Index     │                 │
│   │  (SAR vectors)   │   │  (OPT vectors)   │                │
│   └────────┬─────────┘   └────────┬─────────┘                │
│            │                      │                           │
│            └───────────┬──────────┘                           │
│                        ▼                                      │
│               Top-K Results + Metadata                        │
└──────────────────────────────────────────────────────────────┘
```

---

## Demo

### Text → Image Search
> **Query:** *"dense urban area with high-rise buildings near a coastline"*

The engine tokenizes the query via RemoteCLIP's text encoder, projects it into the 512-D space, and returns the top-5 closest satellite images across both SAR and Optical modalities.

### Image → Image Search (Cross-Modal)
> **Upload:** A SAR image → **Returns:** Matching Optical images (and vice-versa)

The bifurcated routing layer automatically detects the query modality from the filename and restricts results to the **opposite** sensor type for true cross-modal retrieval.

---

## Getting Started

### Prerequisites

- **Python 3.9+** (tested on 3.10/3.11)
- **Git**
- **pip** (comes with Python)
- ~2 GB disk space for the model weights + dataset

### 1. Clone the Repository

```bash
git clone https://github.com/tusharsingh-sde/ASE-Antrikhsa-Search-Engine.git
cd ASE-Antrikhsa-Search-Engine
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** This installs `torch`, `torchvision`, `clip` (from OpenAI), `faiss-cpu`, `streamlit`, `numpy`, `pandas`, and `Pillow`. If you have a CUDA-enabled GPU, you may replace `faiss-cpu` with `faiss-gpu` for hardware-accelerated search.

### 4. Download the RemoteCLIP Model Weights

The application requires **RemoteCLIP ViT-B/32** weights placed at `models/RemoteCLIP-ViT-B-32.pt`.

1. Download the weights from the official [RemoteCLIP repository](https://github.com/ChenDelong1999/RemoteCLIP).
2. Place the `.pt` file inside the `models/` directory:
   ```
   models/
   └── RemoteCLIP-ViT-B-32.pt
   ```

### 5. Download the Dataset

See the [Dataset](#dataset) section below.

---

## Dataset

The dataset is **not included in this repository** due to its large size. It is hosted on **Google Drive** as a ZIP archive.

### Contents of the ZIP

| Item | Description |
|---|---|
| `gallery/` | Folder containing **16,000+ satellite images** (SAR and Optical, `.png` format) |
| `database_names.npy` | NumPy array of filenames corresponding to each indexed vector |
| `database_vectors.npy` | NumPy array of pre-computed 512-D RemoteCLIP feature vectors |

### Step-by-Step Download Instructions

**Step 1 — Download the ZIP file**

Open the following Google Drive link in your browser:

> 📥 **[Download Dataset (Google Drive)](https://drive.google.com/file/d/1nrXonG05ZwLpaSLzCyCR_o5QKN6jChov/view?usp=sharing)**

Click the **Download** button (or the ⬇ icon) to save the ZIP file to your machine.

**Step 2 — Extract into the project root**

Extract the contents of the ZIP file into the **root directory** of this project so that the files are placed as follows:

```bash
# Linux / macOS
unzip <downloaded-file>.zip -d .

# Windows (PowerShell)
Expand-Archive -Path <downloaded-file>.zip -DestinationPath .
```

**Step 3 — Verify the folder structure**

After extraction, your project root should look like this:

```
ASE/
├── dataset/
│   └── gallery/          ← 16K+ SAR & Optical images
├── models/
│   └── RemoteCLIP-ViT-B-32.pt
├── database_names.npy    ← from the ZIP
├── database_vectors.npy  ← from the ZIP
├── app.py
├── build_db.py
├── extractor.py
├── split.py
├── requirements.txt
└── ...
```

> **Important:** The `gallery/` folder must be placed inside `dataset/` (i.e. `dataset/gallery/`). The `.npy` files go in the project root.

---

## Usage

### Run the Application

```bash
streamlit run app.py
```

The app will launch in your default browser at `http://localhost:8501`.

### Rebuild the Vector Database (Optional)

If you modify the gallery (add/remove images), regenerate the index:

```bash
python build_db.py
```

This scans every image in `dataset/gallery/`, computes its 512-D RemoteCLIP embedding, and saves the results to `database_vectors.npy` and `database_names.npy`.

### Prepare a Raw Kaggle Dataset (Optional)

If you downloaded the raw [Paired SAR-Optical Dataset](https://www.kaggle.com/) from Kaggle (where SAR and Optical views are stitched side-by-side in a single image):

1. Update the `raw_folder` path in `split.py` to point to your extracted Kaggle download.
2. Run:
   ```bash
   python split.py
   ```
   This splits each stitched image into separate `_SAR.png` and `_OPT.png` files and saves them into `dataset/gallery/`.

### Verify Model Loading

To quickly verify that RemoteCLIP weights load correctly:

```bash
python test_remote.py
```

Expected output:
```
🚀 Loading RemoteCLIP...
✅ RemoteCLIP Weights Loaded Successfully!
🎯 Success! Vector generated with shape: torch.Size([1, 512])
```

---

## Project Structure

```
ASE-Antrikhsa-Search-Engine/
│
├── app.py                  # Main Streamlit application (UI + search logic)
├── extractor.py            # RemoteCLIP encoder (image & text embeddings)
├── build_db.py             # Offline indexer — builds the FAISS vector database
├── split.py                # Dataset prep — splits paired SAR-Optical images
├── test_remote.py          # Quick sanity check for model loading
├── project_context.txt     # Detailed project documentation
│
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
│
├── .streamlit/
│   └── config.toml         # Streamlit dark theme configuration
│
├── models/                 # (git-ignored) Model weights
│   └── RemoteCLIP-ViT-B-32.pt
│
├── dataset/                # (git-ignored) Image gallery
│   ├── gallery/            #   16K+ SAR & Optical satellite images
│   └── query/              #   Sample query images
│
├── database_vectors.npy    # (git-ignored) Pre-computed 512-D feature vectors
└── database_names.npy      # (git-ignored) Corresponding image filenames
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **ML Model** | [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) (ViT-B/32) |
| **Framework** | [OpenAI CLIP](https://github.com/openai/CLIP) |
| **Vector Search** | [FAISS](https://github.com/facebookresearch/faiss) (IndexFlatL2) |
| **Web UI** | [Streamlit](https://streamlit.io/) |
| **Deep Learning** | [PyTorch](https://pytorch.org/) |
| **Data Handling** | NumPy, Pandas, Pillow |

---

## Contributing

Contributions, issues, and feature requests are welcome!

1. **Fork** the repository.
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`).
3. **Commit** your changes (`git commit -m 'Add amazing feature'`).
4. **Push** to the branch (`git push origin feature/amazing-feature`).
5. **Open** a Pull Request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Tushar Singh**

- 🔗 GitHub: [@tusharsingh-sde](https://github.com/tusharsingh-sde)

---

<p align="center">
  <sub>Built with ❤️ for the remote sensing community</sub>
</p>
