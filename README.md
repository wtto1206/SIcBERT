# SIcBERT: Scalar Implicature classification with DeBERTa

**by [Vittorio Ciccarelli](https://slam.phil.hhu.de/authors/vitto/)**

## Project Description

This repository contains the implementation and evaluation of a fine-tuned **DeBERTa** model for classifying **Scalar Implicatures (SI)**. As part of the *Advanced Python for NLP* seminar, this project focuses on testing the effectiveness of the model using a custom dataset of **Scalar Implicatures** (SIs) involving **scalar expressions**. The model was initially fine-tuned on the **[SIGA](https://github.com/Rashid-Ahmed/SIGA-nli)** dataset and then evaluated on our own dataset to assess its performance.

## Getting Started

To get started with this project, you need to clone the **SIGA** repository first, as it provides the foundation for training and evaluating the model. Follow the instructions from the SIGA repo to set up their environment, fine-tune the model, and prepare it for evaluation on our dataset.

### 1. Clone the **SIGA** repository

```bash
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

```bash
cd experiments
poetry lock
poetry install
```

This will create a **.venv** (virtual environment) with all the required dependencies installed.

### 4. Manually install PyTorch (if using GPU)

By default, **Poetry** installs PyTorch for CPU usage. If you want to use GPU acceleration, install PyTorch manually inside the virtual environment:

```bash
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

(Modify the CUDA version based on your system configuration.)


### 5. Fine-tune the DeBERTa model on the SIGA dataset
Once the environment is set up, fine-tune the DeBERTa model on the SIGA dataset:

(run this inside the `experiments/` folder)

```bash
python cli.py train <output_model_path>
```

This will create a fine-tuned model, which you can then use for evaluation.

### 6. Evaluate the fine-tuned model on the SIGA dataset
To evaluate the model on the SIGA dataset, follow the evaluation instructions from the SIGA repository:

(run this inside the `experiments/` folder)

```bash
python cli.py evaluate
```

(Note: By default, evaluation expects a temp folder in the `experiments` directory that contains the model checkpoint. For further information, refer to `config.py -> ModelConfig -> model_checkpoint` in the SIGA repository.

### 7. Evaluate the fine-tuned model on our custom Scalar Implicature dataset

#### Download and Prepare the Evaluation Dataset

Instead of cloning this repository, you only need some files from it. If you want to evaluate the fine-tuned model on **our** dataset:

1. Download the already processed evaluation dataset from our **data/** folder (`evaluation_dataset.csv`).
2. If needed, you can also find the **original** dataset ([Sun et al., 2024](https://psycnet.apa.org/fulltext/2023-98265-001.html)) in `data/data.csv`.
3. To preprocess `data.csv`, use the `data_processing.py` script located in the `scripts/` folder.

#### Modify SIGA Repository Files

To use `evaluation_dataset.csv`, you need to modify certain files in the **SIGA** repository:

##### 1. Modify `config.py`
   
Navigate to:

```bash
nano ~/NLI_AP/SIGA-nli/experiments/siga_nli/config.py
```

Replace the `DataConfig` class with:

```python
class DataConfig(BaseModel):
    train_data_dir: Path = Path.cwd().parent / Path("data") / "train_dataset.csv"
    eval_data_dir: str = Path.cwd().parent / Path("data") / "evaluation_dataset.csv"
    # id_test_data_dir: str = Path.cwd().parent / Path("data") / "test_id_dataset.csv"
    # ood_test_data_dir: str = Path.cwd().parent / Path("data") / "test_ood_dataset.csv"
    max_token_length: Optional[int] = 256
    padding: bool = True
    truncation: bool = True
    num_labels: int = 3
```

##### 2. Modify `evaluate.py`

Navigate to:

```bash
nano ~/NLI_AP/SIGA-nli/experiments/siga_nli/evaluate.py
```

Modify the `evaluate` function:

```python
def evaluate(config: Config):
    auto_config, tokenizer = initialize_tokenizer(config)
    model = load_model(config, auto_config)
    eval_dataset, _ = load_data(config.data.eval_data_dir, first_split=1)
    evaluation_results = evaluate_model(eval_dataset, model, tokenizer, config.training.evaluation_metric, config)
    logger.info(evaluation_results[config.training.evaluation_metric])
```
##### 3. Add the evaluation dataset

Move `evaluation_dataset.csv` (downloaded from our repository) into the SIGA repository's `data/` folder:

```bash
mv /path/to/evaluation_dataset.csv ~/NLI_AP/SIGA-nli/data/
```

##### 4. Run the Evaluation
Now, you can evaluate the fine-tuned model on our dataset:

(run this inside the `experiments/` folder)

```bash
python cli.py evaluate
```

### 8. Compare with the Non-Fine-Tuned Model
If you also want to compare these results with a non-fine-tuned version of the model, you can use the provided notebook `eval_not_finetuned.ipynb` in the `notebooks/` folder in our repository:

```bash
notebooks/eval_not_finetuned.ipynb
```

### License
All source code is made available under the `MIT License`. You are free to use, modify, and distribute the code, provided that you give appropriate credit to the authors. See LICENSE.md for the full license text.

### Acknowledgements
This project relies on the [SIGA repository](https://github.com/Rashid-Ahmed/SIGA-nli) for fine-tuning the DeBERTa model. Their dataset and code were critical for the development of this project. You can find the original SIGA repository here:

*SIGA: A Naturalistic NLI Dataset of English Scalar Implicatures with Gradable Adjectives*

## Contact
If you have any questions about this project, feel free to contact me at Vittorio.Ciccarelli@hhu.de.


