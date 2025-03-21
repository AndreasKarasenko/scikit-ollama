# Installation and setup

This project depends on [Scikit-LLM](https://github.com/iryna-kondr/scikit-llm) and [Ollama](https://ollama.com/), both will be installed through pip, but you will need to do some additional setup.

First you should install Ollama for your OS (e.g. Linux):

```
curl -fsSL https://ollama.com/install.sh | sh
```

This will download and setup Ollama to run on your local machine and allow you to load and manage several Large Language Models (LLM).

After this you can configure and start the server, pull models, run chat instances or use the library. To configure the server please see the environment variables from [Ollama](https://github.com/ollama/ollama/blob/80ee9b5e47fc0ea99d1f3f33224923266627c15c/envconfig/config.go) and consult the [FAQ](https://github.com/ollama/ollama/blob/main/docs/faq.md).

```bash
export OLLAMA_MAX_LOADED_MODELS=3 # sets the max number of loaded models
export OLLAMA_NUM_PARALLEL=2 # sets the max number of parallel tasks
```

To serve, pull and run models:

```bash
ollama serve # starts the server
ollama pull llama3:8b # downloads llama3 in the 8b configuration
ollama pull nomic-embed-text # a typical embedding model
ollama run llama3:8b # opens a chat instance with llam3:8b as the model
```

Stopping the Ollama server is less straigthforward unfortunately.
You will have to find the process ID (PID) using `pgrep` and kill the task.

```bash
pgrep ollama # returns a PID
kill <PID> # kill the PID
```
