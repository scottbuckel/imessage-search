# iMessage Chat Log AI Assistant

This Python script implements a Retrieval-Augmented Generation (RAG) system that allows you to ask questions about your iMessage chat logs stored in JSON format. It uses semantic search to find relevant parts of your conversations and then leverages a local Large Language Model (LLM) via Ollama to generate answers.

## Description

The script processes JSON files containing your iMessage chat history. It performs the following steps:

1.  **Reads and Chunks Data:** Loads chat messages from JSON files and intelligently segments them into meaningful chunks based on semantic similarity.
2.  **Embeds Chunks:** Uses the `sentence-transformers` library to create vector embeddings of these chunks.
3.  **Builds Search Index:** Creates a Faiss index for efficient similarity search over the embeddings.
4.  **Caches Data:** Stores the embeddings, Faiss index, processed chunks, and file states in a cache directory to speed up subsequent runs.
5.  **Answers Questions:** When you ask a question, the script:
    * Embeds your query.
    * Searches the Faiss index for relevant message chunks.
    * Formats a prompt containing the relevant context and your question.
    * Sends the prompt to a local Ollama instance running the `mistral` model.
    * Prints the generated answer.

## Prerequisites

Before running this script, you need to have the following installed:

* **Python 3.6 or higher:** You likely have this already if you're using a modern operating system.
* **pip:** Python package installer.
* **Ollama:** A tool for running large language models locally. You can find installation instructions on the [Ollama website](https://ollama.ai/). Make sure you have the `mistral` model downloaded in Ollama by running `ollama pull mistral` in your terminal.
* **Python Libraries:** Install the required libraries using pip:
    ```bash
    pip install sentence-transformers faiss-cpu numpy scikit-learn
    ```

## Installation

1.  **Save the script:** Save the provided Python code as a `.py` file (e.g., `imessage_ai.py`).

## Configuration

The script has several configuration variables at the beginning that you might want to adjust:

* `INPUT_PATH`: **REQUIRED** - Set this to the path of the directory containing your iMessage chat log JSON files. If you have a single JSON file, you can also point this directly to that file.
* `CACHE_DIR`: Path to the directory where the script will store cached data (embeddings, index, etc.). It will be created if it doesn't exist.
* `EMBEDDINGS_CACHE_FILE`, `INDEX_CACHE_FILE`, `FILE_STATE_CACHE_FILE`, `CHUNKS_CACHE_FILE`: These define the specific filenames for the cached data within the `CACHE_DIR`. You usually don't need to change these.
* `RELEVANT_CHUNK_SIZE`: Determines how many relevant message chunks the script will retrieve to answer your question.
* `MIN_CHUNK_LINES`: The minimum number of messages required for a chunk. Shorter sequences will be merged.
* `MAX_CHUNK_LINES`: The hard limit on the number of messages in a single chunk. Longer sequences will be split.
* `BATCH_SIZE`: The number of JSON files to process in each batch during the initial indexing or rebuilding.
* `SIMILARITY_DROP_THRESHOLD`: A value between 0 and 1 that determines when a significant drop in semantic similarity occurs, indicating a potential new chunk. Lower values might result in more chunks.
* `SIMILARITY_LOOKAHEAD`: The number of subsequent messages to consider when calculating the semantic similarity drop.

**To configure these variables:** Open the `imessage_ai.py` file in a text editor and modify the values of the variables at the top of the script according to your needs. **Make sure to set the `INPUT_PATH` correctly!**

## Usage

1.  **Open your terminal or command prompt.**
2.  **Navigate to the directory where you saved the `imessage_ai.py` file.**
3.  **Run the script:**
    ```bash
    python imessage_ai.py
    ```
4.  **The script will initialize:** It will load the SentenceTransformer model, check for cached data, process any new or modified files, and build or load the search index. This might take some time on the first run, especially if you have a lot of chat logs.
5.  **Ask questions:** Once the initialization is complete, you will see a prompt:
    ```
    Ask a question about your texts (or type 'exit' to quit):
    ```
    Type your question about your iMessage conversations and press Enter. The script will search for relevant information and generate an answer using Ollama.
6.  **Continue asking questions:** You can keep asking questions until you type `exit` and press Enter.

## Directory Structure (Example)

your_project_directory/
├── imessage_ai.py
├── texts/                      # Example input directory
│   ├── chat_with_john.json
│   ├── group_chat_family.json
│   └── ...
└── cache/                      # Cache directory created by the script
├── transformers_cache/
├── chunk_embeddings.pkl   # (Potentially older format)
├── chunk_faiss_index.bin
├── chunk_file_state.json
└── all_chunks_cache.pkl


## Troubleshooting

* **`Error: 'ollama' command not found.`:** Make sure Ollama is installed correctly and that the `ollama` command is in your system's PATH environment variable.
* **`Fatal Error: Could not load SentenceTransformer model:`:** Ensure you have installed the `sentence-transformers` library (`pip install sentence-transformers`) and that you have a stable internet connection during the first run (to download the model if it's not cached).
* **No relevant chunks found:** Try rephrasing your question or adjusting the `RELEVANT_CHUNK_SIZE` or `SIMILARITY_DROP_THRESHOLD` configuration variables.
* **Ollama timeout:** If your questions require complex reasoning or the retrieved context is very large, Ollama might take longer to respond. You can try increasing the timeout in the `generate_answer` function if needed (though the default is already set to 90 seconds).
* **Cache inconsistencies:** If you encounter issues after modifying your input files, you can try deleting the contents of the `CACHE_DIR` to force a rebuild of the index and chunks.

This README should provide a good starting point for anyone wanting to use your iMessage chat log AI assistant. Remember to adapt it further if you add more features or have specific instructions for your users.