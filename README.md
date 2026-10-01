# HW4 starter: a tiny text generator

CPSC 1710 · Homework 4

`tiny_rnn.py` trains a very small recurrent neural network on ten short sentences, then generates text one character at a time. Choose either method below. Both run the same Python file.

## Method 1: run in Google Colab

This is the quickest route and does not install anything on your computer.

[Open the starter in Colab](https://colab.research.google.com/github/cpsc1710/hw4-starter/blob/main/run-in-colab.ipynb)

1. Open the link above.
2. Choose **Runtime → Run all**.
3. Wait for training to finish. TensorFlow may print a few setup messages first.
4. Look for generated text at four different temperatures.

Colab runs on Google's computers. Files created or changed there are temporary unless you download them or save a copy of the notebook to Drive.

## Method 2: run locally

This route is useful if you want to work in VS Code and learn how a Python environment keeps one project's packages separate from another's.

### 1. Get the files

Use **Code → Download ZIP** on this GitHub page and unzip it, or clone the repository:

```bash
git clone https://github.com/cpsc1710/hw4-starter.git
cd hw4-starter
```

Open the resulting folder in VS Code.

### 2. Check Python

Python 3.12 is the safest choice across macOS and Windows for this project.

```bash
python3 --version
```

On Windows, use `py --version`. If you do not have Python 3.12, install it from [python.org](https://www.python.org/downloads/).

### 3. Create and activate a project environment

macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

When the environment is active, your terminal prompt normally begins with `(.venv)`.

Confirm which Python will run:

```bash
python -c "import sys; print(sys.executable)"
```

The printed path should contain this project's `.venv` folder. This check is more useful than memorizing activation commands: it tells you—and any coding agent—exactly which Python is in use.

### 4. Install the packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow is a large download, so installation may take a few minutes.

### 5. Run the starter

```bash
python tiny_rnn.py
```

The program prints the corpus, vocabulary, final training loss, and generated text at four temperatures. Before each sample, it also shows the five most likely choices for the first generated character.

The starter uses `SEED = 1710` so everyone can compare the same first run. Change that number—or set it to `None`—to see how random initialization and sampling change the result.

## If local setup gets stuck

Use the Colab method and keep moving. Local environment setup is useful practice, but it is not the learning objective of the RNN exercise.
