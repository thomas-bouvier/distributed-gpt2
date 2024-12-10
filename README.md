# distributed-gpt2

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [nanoGPT](https://github.com/karpathy/nanoGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

| Model  | Architecture                                                                      | Max Parameter Count | Training Data                                                                                                               |
|--------|-----------------------------------------------------------------------------------|---------------------|-----------------------------------------------------------------------------------------------------------------------------|
| GPT-1  | 12-level, 12-headed Transformer decoder (no encoder), followed by linear-softmax. | 0.12 billion        | BookCorpus: 4.5 GB of text, from 7000 unpublished books of various genres.                                                  |
| GPT-2  | GPT-1, but with modified normalization                                            | 1.5 billion         | WebText: 40 GB of text, 8 million documents, from 45 million webpages upvoted on Reddit.                                    |
| GPT-3  | GPT-2, but with modification to allow larger scaling.                             | 175 billion         | 570 GB plaintext, 300 billion tokens of CommonCrawl, WebText, English Wikipedia, and two books corpora (Books1 and Books2). |

## Setup

Follow the instructions from [`devenv`](https://devenv.sh/getting-started/)
using the instructions found here.

### Install `nix`

```bash
### Via https://zero-to-nix.com/start/install (recommended)
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install

### Via https://devenv.sh/getting-started/
## Linux
sh <(curl -L https://nixos.org/nix/install) --daemon

## macOS
curl -L https://raw.githubusercontent.com/NixOS/experimental-nix-installer/main/nix-installer.sh | sh -s install

## WSL2
sh <(curl -L https://nixos.org/nix/install) --no-daemon
```

### Install `devenv`

```bash
## General
nix-env -iA devenv -f https://github.com/NixOS/nixpkgs/tarball/nixpkgs-unstable

## NixOS
# Add the following to your configuration.nix somewhere
environment.systemPackages = [ 
  pkgs.devenv
];
```

#### `devenv.nix`

Defines the configuration for the `devenv` shell. This is where we define all
the tooling, packages, scripts, services, processes, etc. that we need for the
project.

#### `devenv.yaml`

The `yaml` defines the sources for all the packages, i.e. where are we getting
the cached builds or build instructions for `nix`.

#### TODO

CUDA support:

- https://github.com/johnrizzo1/myada/blob/6928288910bfd1df8993d8c61bdc5d24d92b4c9e/devenv.nix
- https://github.com/borh/dm-annotations/blob/06035f4547c68b7bf03b757215b48c76568d8d15/devenv.nix
