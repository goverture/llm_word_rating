# Crossword Word Rating

Evaluate the quality of words for crossword puzzles using a language model. This setup leverages vLLM for offline batched inference on a Lambda Labs instance with an A100 GPU.

## Wordlist

The original wordlist (`wordlist.txt`) was downloaded from [Crossword Nexus](https://crosswordnexus.com/).

## Setup Instructions

### 1. Install Python 3.12.1 with pyenv

Ensure your system is updated and has the necessary dependencies:

```sh
sudo apt update
sudo apt install -y \
  make build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses5-dev libncursesw5-dev xz-utils tk-dev \
  libffi-dev liblzma-dev git
```

#### Install pyenv

```sh
curl https://pyenv.run | bash
```

Update your shell configuration:

```sh
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc  # Use ~/.zshrc if using Zsh
```

#### Install and set Python 3.12.1

```sh
pyenv install 3.12.1
pyenv global 3.12.1
```

Verify the installation:

```sh
python --version
```

### 2. Install vLLM

Install vLLM (version 0.7.0 at this time):

```sh
pip install vllm
```

## Running the Evaluation

Once the setup is complete, you can run the evaluation script to process the words in `wordlist.txt` using vLLM.

