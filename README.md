# ollama-rag

## Installation

```sh
task install
```

## Usage

### To load a database with documents

```sh
python main.py load --documents-path ~/notes --database-path ~/vector_db --database-name notes
```

### To ask a question based on a loaded database

```sh
python main.py ask --database-path ~/vector_db --database-name notes --model-name llama3 --question "How to perform a domian transfer?"
```


