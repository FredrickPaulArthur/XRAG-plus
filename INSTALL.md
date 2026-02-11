# Installation — XRAG-plus

This document shows how to install **XRAG-plus** for development and evaluation.


## Prerequisites

- Git **>= 2.25**
- Python **3.13.2**
- `pip` **>= 26.0** and `virtualenv`
- *(Optional, GPU)* CUDA-compatible drivers if you plan to run heavy models locally

---

## System Packages (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  git \
  python3-venv \
  python3-dev
```

---

## Clone Repo and Create venv

```bash
git clone https://github.com/FredrickPaulArthur/XRAG-plus.git
cd XRAG-plus

python3 -m venv _venv
./_venv/Scripts/activate
```

---

## Install Python Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## GPU-Enabled PyTorch

Install PyTorch with CUDA following the below command.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

---