### Quantifying Motor Characteristics in Parkinson's Disease Using Computer Vision Techniques

This repository contains the official implementation of our paper:  
**[Interpretable and Granular Video-Based Quantification of Motor Characteristics from the Finger Tapping Test in Parkinson's Disease](https://doi.org/10.1038/s41531-026-01307-w).**

Tahereh Zarrat Ehsan, Michael Tangermann, Yağmur Güçlütürk, S. Shin, K. C. Ho, Bastiaan R. Bloem, Luc J. W. Evers  
Radboud University, Donders Institute for Brain, Cognition and Behaviour

**This repository was adapted by Emma Luisa Lakin for her AI Bachelor Thesis.**
---

## 📂 Repository Structure

```
📁 VideoBased-PD-Biomarkers/
│
├── 📁 data/
│ ├── 📁 raw/ # Raw data and extracted keypoints (.pkl)
│ └── 📁 processed/ # Processed CSVs and derived feature files
│
├── 📁 src/
│ ├── 📁 preprocessing/
│ │ └── 📄 keypoint_extraction.py # Extracts hand keypoints using Mediapipe
│ ├── 📁 feature_extraction/
│ │ └── 📄 feature_extraction.py # Computes motor features (amplitude, speed, etc.)
│ ├── 📁 helper_functions/
│ │ └── 📄 models_functions.py
│ │ └── 📄 helper_functions.py
│ │ └── 📄 helper_plot_functions.py
│ ├── 📁 model_analysis_notebooks/
│ │ └── 📄 handedness.ipynb
│ │ └── 📄 model_evaluation.ipynb
│ │ └── 📄 models.ipynb
│ │ └── 📄 pareto_fronts.ipynb
│ │ └── 📄 raw_signal.ipynb
│ │ └── 📄 sweep_number_comparison.ipynb
│ │ └── 📄 sweep_type_comparison.ipynb
│ │ └── 📄 updrs_vs_trial5.ipynb
│ │ └── 📄 variability.ipynb
│ │ └── 📄 visualisation.ipynb
├── 📄 environment.yml # Conda environment setup
└── 📘 README.md # Project documentation
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/TaherehZarratEhsan/VideoBased-PD-Biomarkers.git
cd VideoBased-PD-Biomarkers
```

### Conda
```bash
conda env create -f environment.yml
conda activate parkinson-digital-biomarkers
```

---
## ▶️ Usage
### 🔹 Part 1: Demo (video => motor quanitification)

This demo offers a simple end-to-end example that runs the full analysis pipeline — from a raw video to automatic feature extraction and visualization.
It can be executed locally without any dataset setup or preprocessing steps from the main implementation.

📁 Script location
    ```
src/demo/ft_video_analysis.py
    ```

▶️ Run the demo
```bash
python ft_video_analysis.py --video_path "C:/Users/Tahereh/video.MP4" --hand2track Right
```
Arguments:
    ```
--video_path: Path to the input video file
--hand2track: Which hand to analyze (Left or Right)
    ```
### 🔹 Part 2: Official Implementation
#### 🔹 Keypoint Extraction

If you want to build your own pickle file (`video_keypoints.pkl`) from raw videos, first prepare a CSV file with the following columns:

- **video_path**: Full path to each video  
- **score**: Clinical MDS‑UPDRS score (put 0 here, as this is not used in the achelor thesis project
- **id**: Patient ID  

Save it in `data/raw/sat_finger_tapping.csv`.

The Mediapipe hand landmark model is required to extract keypoints.  
It will be automatically downloaded on first run from:

[Google Cloud Storage – Mediapipe Hand Landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

Alternatively, you can download it manually. After download, the file should be located at:

```
src/preprocessing/hand_landmarker.task
```

Then run:

```bash
python src/preprocessing/keypoint_extraction.py
```

This will:
- Process all listed videos using Mediapipe’s HandLandmarker  
- Extract distance‑based signals 
- Save a dictionary with `video_path`, `distance signal`, `keypoints`, `id`, `label`, and `fps`  
- craete Participant folders with the name structure 00X in `data/raw`

  
The output will be stored in:

```
data/raw/video_keypoints.pkl
```
#### 🔹 Feature Extraction

After downloading keypoints (or generating) and placing `video_keypoints.pkl` in `data/raw/`, run:

```bash
python src/feature_extraction/feature_extraction.py
```

This will extract motor features (amplitude, speed, cycle duration, etc.) and generate:

```
data/processed/finger_tapping_features.csv
```
#### 🔹 Feature Analysis for SAT in Finger Tapping
After adjusting paths to the relevant .csv/.pkl files in the helper functions, the Juypter notebooks can be run for visualisation purposes.
    
## 📥 Data Access

Data from the [Personalized Parkinson Project](https://www.personalizedparkinsonproject.com/home) used in the present study were retrieved from the [PEP database](https://pep.cs.ru.nl/index.html).  
The PPP data are available upon request via [ppp-data@radboudumc.nl](mailto:ppp-data@radboudumc.nl).  
More details on the procedure can be found on the [project website](https://www.personalizedparkinsonproject.com/home).

---

## 📚 Citation

If you use this repository in your research, please cite:

> Zarrat Ehsan, T., Tangermann, M., Güçlütürk, Y., Shin, S., Ho, K. C., Bloem, B. R., & Evers, L. J. W.  
> *Interpretable and Granular Video-Based Quantification of Motor Characteristics from the Finger-Tapping Test in Parkinson’s Disease.*  
> **npj Parkinson’s Disease**, 2026.  
> https://doi.org/10.1038/s41531-026-01307-w

---

## 📘 BibTeX

```bibtex
@article{zarratehsan2026finger,
  title = {Interpretable and Granular Video-Based Quantification of Motor Characteristics from the Finger-Tapping Test in Parkinson's Disease},
  author = {Zarrat Ehsan, Tahereh and Tangermann, Michael and G{\"u}{\c{c}}l{\"u}t{\"u}rk, Ya{\u{g}}mur and Shin, S. and Ho, K. C. and Bloem, Bastiaan R. and Evers, Luc J. W.},
  journal = {npj Parkinson's Disease},
  year = {2026},
  doi = {10.1038/s41531-026-01307-w}
}
```

---

## 📜 License

This project is licensed under the Apache 2.0 License.
