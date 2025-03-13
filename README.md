# SIcBERT: Fine-Tuning roBERTa for Scalar Implicature Classification

**by [Vittorio Ciccarelli](https://slam.phil.hhu.de/authors/vitto/)**

## Project Description

This repository contains the implementation and evaluation of a fine-tuned **roBERTa** model for classifying **Scalar Implicatures (SI)**. As part of the *Advanced Python for NLP* seminar, this project focuses on testing the effectiveness of the model using a custom dataset of **Scalar Implicatures** (SIs) involving **gradable adjectives**. The model was initially fine-tuned on the **SIGA** dataset and then evaluated on our own dataset to assess its performance.

## Getting Started

To get started with this project, you need to clone the **SIGA** repository first, as it provides the foundation for training and evaluating the model. Follow the instructions from the SIGA repo to set up their environment, fine-tune the model, and prepare it for evaluation on our dataset.

### 1. Clone the **SIGA** repository

```
git clone https://github.com/username/SIGA.git
cd SIGA
```

### 2. Create a Virtual Environment

It is recommended to create a virtual environment before installing dependencies. You can use **venv** as follows:

```bash
python3 -m venv <name_of_venv>
source <name_of_venv>/bin/activate
```

Make sure to substitute `<name_of_venv>` with an actual name for your environment.

### 3. Install dependencies for SIGA

Ensure you have **Python 3.9.6** and **Poetry** installed on your system before proceeding.

- Install [Poetry](https://python-poetry.org/docs/#installation) if you haven't already.
- Navigate to the **`experiments/`** directory inside the SIGA repository and run:

```
cd experiments
poetry lock
poetry install
```

This will create a **.venv** (virtual environment) with all the required dependencies installed.

### 4. Manually install PyTorch (if using GPU)

By default, **Poetry** installs PyTorch for CPU usage. If you want to use GPU acceleration, install PyTorch manually inside the virtual environment:

```
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

(Modify the CUDA version based on your system configuration.)


### 5. Fine-tune the roBERTa model on the SIGA dataset
Once the environment is set up, fine-tune the roBERTa model on the SIGA dataset:

```
python cli.py train <output_model_path>
```

This will create a fine-tuned model, which you can then use for evaluation.

### 6. Clone the SIcBERT repository
Next, clone this repository to start evaluating the fine-tuned model:

```
git clone https://github.com/yourusername/SIcBERT.git
cd SIcBERT
```

### 7. Install the required dependencies for this project
If you haven’t already, install the necessary dependencies for the SIcBERT repository:

```
pip install -r requirements.txt
```

### 8. Evaluate the fine-tuned model on our custom Scalar Implicature dataset
Once the fine-tuned model is available, use it to evaluate on the custom dataset for Scalar Implicatures:

```
python evaluate.py --model_path <path_to_finetuned_model>
```

Replace <path_to_finetuned_model> with the path to the model you fine-tuned earlier.

### Data
Our custom dataset consists of Scalar Implicatures that include gradable adjectives. It was used to evaluate the performance of the fine-tuned roBERTa model. The dataset is included in the `data/` folder.

### Preprocessing the Data
Ensure the data is in the correct format before evaluating the model. You can run the preprocessing script if needed:

```
python preprocess.py
```

### License
All source code is made available under the `MIT License`. You are free to use, modify, and distribute the code, provided that you give appropriate credit to the authors. See LICENSE.md for the full license text.

### Acknowledgements
This project relies on the `SIGA repository`LINK for fine-tuning the roBERTa model. Their dataset and code were critical for the development of this project. You can find the original SIGA repository here:

*SIGA: A Naturalistic NLI Dataset of English Scalar Implicatures with Gradable Adjectives*

## Contact
If you have any questions about this project, feel free to contact me at vittorio.ciccarelli@example.com.


