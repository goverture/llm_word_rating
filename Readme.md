# Crossword word rating

Evaluate the quality of words with an LLM.
Use vLLM offline batched inference to evaluate each word in a word list.

Steps to run on a lambdalab instance (A100)

## Wordlist

The original wordlist (wordlist.txt) file was downloaded from [Crossword Nexus](https://crosswordnexus.com/)

## Install python 3.12.1 with pyenv

```sh
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
git

# Install pyenv
curl https://pyenv.run | bash

# Configure your shell
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
# Then, reload the shell:
source ~/.bashrc  # Or ~/.zshrc

# Install Python with pyenv
pyenv install 3.12.1  # Replace with the latest version if needed
pyenv global 3.12.1   # Set it as the default version

# Verify
python --version
```

## Install vllm

(v0.7.0 at this time)
```
pip install vllm
```
