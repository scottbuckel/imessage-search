import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
import subprocess
import pprint
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# --- Configuration ---
INPUT_PATH = '/Volumes/ScottsT7/imessage_ai/texts/llm_texts_json/'  # Path to your JSON files
CACHE_DIR = '/Volumes/ScottsT7/imessage_ai/cache/transformers_cache'
EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, 'chunk_embeddings.pkl') # Note: This seems unused now, Faiss index is primary
INDEX_CACHE_FILE = os.path.join(CACHE_DIR, 'chunk_faiss_index.bin')
FILE_STATE_CACHE_FILE = os.path.join(CACHE_DIR, 'chunk_file_state.json')
CHUNKS_CACHE_FILE = os.path.join(CACHE_DIR, 'all_chunks_cache.pkl') # Cache for the actual chunk data

RELEVANT_CHUNK_SIZE = 5  # How many chunks to retrieve for answering
MIN_CHUNK_LINES = 4     # Minimum messages per chunk
MAX_CHUNK_LINES = 50    # Maximum messages per chunk (hard limit)
BATCH_SIZE = 10         # How many files to process in one go
SIMILARITY_DROP_THRESHOLD = 0.88 # Threshold for semantic split
SIMILARITY_LOOKAHEAD = 2         # How many messages ahead to look for similarity drop

# --- Helper Functions ---

def get_file_state(path):
    """Gets modification time and size for JSON files in a path."""
    file_state = {}
    os.makedirs(os.path.dirname(path), exist_ok=True) # Ensure directory exists if path is a file path
    if os.path.isdir(path):
        try:
            for filename in os.listdir(path):
                if filename.endswith(".json"):
                    filepath = os.path.join(path, filename)
                    try:
                        file_state[filename] = {
                            'modified': os.path.getmtime(filepath),
                            'size': os.path.getsize(filepath)
                        }
                    except OSError as e:
                        print(f"Warning: Could not access file {filepath}: {e}")
        except FileNotFoundError:
             print(f"Warning: Input directory not found: {path}")
             return {}
    elif os.path.isfile(path) and path.endswith(".json"):
        try:
            file_state[os.path.basename(path)] = {
                'modified': os.path.getmtime(path),
                'size': os.path.getsize(path)
            }
        except OSError as e:
             print(f"Warning: Could not access file {path}: {e}")
             return {}
    else:
        print(f"Error: Path '{path}' is not a valid json file or directory containing json files.")
        return {}
    return file_state

def load_cached_index():
    """Loads the Faiss index from cache."""
    if not os.path.exists(INDEX_CACHE_FILE):
        print("No existing Faiss index cache file found. Creating a new one.")
        return None
    try:
        index = faiss.read_index(INDEX_CACHE_FILE)
        print(f"Loaded existing Faiss index from cache ({INDEX_CACHE_FILE}).")
        return index
    except Exception as e:
        print(f"Error loading Faiss index from {INDEX_CACHE_FILE}: {e}. Attempting to rebuild.")
        # Optionally remove corrupted index file
        try:
            os.remove(INDEX_CACHE_FILE)
            print(f"Removed potentially corrupted index file: {INDEX_CACHE_FILE}")
        except OSError:
            pass # Ignore if removal fails
        return None

def save_cached_index(index):
    """Saves the Faiss index to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        faiss.write_index(index, INDEX_CACHE_FILE)
        print(f"Saved Faiss index to cache ({INDEX_CACHE_FILE}).")
    except Exception as e:
        print(f"Error saving Faiss index to {INDEX_CACHE_FILE}: {e}")

def load_cached_chunks():
    """Loads the list of all_chunks from cache."""
    if not os.path.exists(CHUNKS_CACHE_FILE):
        print("No existing chunk cache file found.")
        return []
    try:
        with open(CHUNKS_CACHE_FILE, 'rb') as f:
            all_chunks = pickle.load(f)
        print(f"Loaded {len(all_chunks)} chunks from cache ({CHUNKS_CACHE_FILE}).")
        return all_chunks
    except Exception as e:
        print(f"Error loading chunks cache from {CHUNKS_CACHE_FILE}: {e}. Starting with empty chunks.")
         # Optionally remove corrupted chunk cache file
        try:
            os.remove(CHUNKS_CACHE_FILE)
            print(f"Removed potentially corrupted chunk cache file: {CHUNKS_CACHE_FILE}")
        except OSError:
            pass # Ignore if removal fails
        return []

def save_cached_chunks(all_chunks):
    """Saves the list of all_chunks to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not all_chunks:
        print("Warning: No chunks generated, not saving chunks cache.")
        # Remove old cache file if it exists and no chunks are present
        if os.path.exists(CHUNKS_CACHE_FILE):
            try:
                os.remove(CHUNKS_CACHE_FILE)
                print(f"Removed empty chunk cache file: {CHUNKS_CACHE_FILE}")
            except OSError as e:
                print(f"Warning: Could not remove empty chunk cache file: {e}")
        return

    try:
        with open(CHUNKS_CACHE_FILE, 'wb') as f:
            pickle.dump(all_chunks, f)
        print(f"Saved {len(all_chunks)} chunks to cache ({CHUNKS_CACHE_FILE}).")
    except Exception as e:
        print(f"Error saving chunks cache to {CHUNKS_CACHE_FILE}: {e}")

def load_cached_file_state():
    """Loads the cached file state."""
    if not os.path.exists(FILE_STATE_CACHE_FILE):
        print("No existing file state cache found.")
        return {}
    try:
        with open(FILE_STATE_CACHE_FILE, 'r') as f:
            cached_state = json.load(f)
        print("Loaded file state from cache.")
        return cached_state
    except Exception as e:
        print(f"Error loading file state cache: {e}. Assuming no prior state.")
        return {}

def save_cached_file_state(current_file_state):
    """Saves the current file state."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(FILE_STATE_CACHE_FILE, 'w') as f:
            json.dump(current_file_state, f, indent=4)
        print("Saved file state to cache.")
    except Exception as e:
        print(f"Error saving file state: {e}")

# --- Core Logic ---

def segment_by_similarity_drop_sorted(messages, embedding_model, min_chunk_lines=MIN_CHUNK_LINES, max_chunk_lines=MAX_CHUNK_LINES, lookahead=SIMILARITY_LOOKAHEAD, drop_threshold=SIMILARITY_DROP_THRESHOLD):
    """Segments messages based on cosine similarity drops."""
    n_messages = len(messages)
    if n_messages == 0:
        print("Warning: segment_by_similarity called with zero messages.")
        return [], [] # Return empty lists if no messages
    if n_messages <= min_chunk_lines:
        return [messages], [] # Return the whole thing as one chunk if too short

    # Ensure all messages are strings
    message_texts = []
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and 'message' in msg and isinstance(msg['message'], str):
             message_texts.append(msg['message'])
        else:
            print(f"Warning: Invalid message format at index {i}. Skipping. Content: {msg}")
            # Decide how to handle invalid messages: skip, replace, error out
            # Skipping for now:
            # return [], [] # Or potentially raise an error
            message_texts.append("") # Replace with empty string to keep indices aligned, maybe filter later

    if not message_texts: # If all messages were invalid
         print("Warning: No valid message texts found for segmentation.")
         return [], []

    try:
        embeddings = embedding_model.encode(message_texts)
    except Exception as e:
        print(f"Error encoding messages during segmentation: {e}")
        # Fallback: maybe return a single chunk or raise error
        return [messages], [] # Simple fallback

    similarity_matrix = cosine_similarity(embeddings)
    drop_points = []
    debug_info = []

    for i in range(1, n_messages):
        # Check bounds for similarity matrix access
        if i >= similarity_matrix.shape[0] or i-1 >= similarity_matrix.shape[0]:
             print(f"Warning: Index out of bounds ({i}) for similarity matrix (shape {similarity_matrix.shape}). Skipping drop point calc.")
             continue

        immediate_similarity = similarity_matrix[i - 1][i]
        immediate_drop = 1.0 - immediate_similarity
        total_drop = immediate_drop
        lookahead_similarities = []
        valid_lookaheads = 0

        for j in range(1, lookahead + 1):
            if i + j < n_messages and i + j < similarity_matrix.shape[1] and i - 1 < similarity_matrix.shape[0]:
                lookahead_similarity = similarity_matrix[i - 1][i + j]
                lookahead_drop = 1.0 - lookahead_similarity
                total_drop += lookahead_drop
                lookahead_similarities.append(f"L+{j}: {lookahead_similarity:.4f} (Drop: {lookahead_drop:.4f})")
                valid_lookaheads += 1
            else:
                 # Optional: Log if lookahead goes out of bounds
                 # print(f"Debug: Lookahead {j} for index {i} out of bounds.")
                 pass


        # Average drop over available points (immediate + valid lookaheads)
        average_drop = total_drop / (1 + valid_lookaheads)

        drop_points.append({
            'index': i,
            'immediate_drop': immediate_drop,
            'average_drop': average_drop,
            'immediate_similarity': immediate_similarity,
            'lookahead_similarities': ", ".join(lookahead_similarities)
        })
        # Reduce debug verbosity if needed
        # debug_info.append({
        #     'index': i, 'type': 'similarity',
        #     'message': f"Sim[{i-1},{i}]: {immediate_similarity:.4f}, ImmDrop: {immediate_drop:.4f}, AvgDrop(L={lookahead}): {average_drop:.4f}, Lookahead: [{', '.join(lookahead_similarities)}]"
        # })

    # Sort drop points by the average magnitude of the drop in descending order
    sorted_drop_points = sorted(drop_points, key=lambda x: x['average_drop'], reverse=True)
    # debug_info.append({'index': -1, 'type': 'sort_info', 'message': f"\nSorted Drop Points (Avg Drop, L={lookahead}):"})
    # for point in sorted_drop_points:
    #     debug_info.append({'index': point['index'], 'type': 'sort_detail', 'message': f"Idx: {point['index']}, AvgDrop: {point['average_drop']:.4f}, ImmDrop: {point['immediate_drop']:.4f}"})

    potential_split_indices = set()
    # debug_info.append({'index': -1, 'type': 'split_consideration', 'message': "\nConsidering Splits:"})

    # This logic determines splits based on sorted high-drop points
    # Ensuring minimum distance between splits can be complex.
    # Let's use a simpler approach: iterate through messages and split if drop threshold is met *and* min chunk size is satisfied since the *last* split.
    split_indices = [0] # Start with the beginning
    last_split = 0
    for i in range(1, n_messages):
        # Find the drop point info for the current index 'i'
        current_drop_info = next((dp for dp in drop_points if dp['index'] == i), None)
        if current_drop_info:
            avg_drop = current_drop_info['average_drop']
            # Check if potential split point meets criteria
            if avg_drop > drop_threshold and (i - last_split) >= min_chunk_lines:
                 # Check if remaining messages also meet min length, prevents tiny last chunk
                 if (n_messages - i) >= min_chunk_lines or (n_messages - i) == 0 : # Allow split if it's the very end
                    split_indices.append(i)
                    last_split = i
                    # debug_info.append({'index': i, 'type': 'split_decision', 'message': f"  Idx {i}: Split (AvgDrop: {avg_drop:.4f} > {drop_threshold}, Len: {i - last_split})"})
                 # else:
                    # debug_info.append({'index': i, 'type': 'split_decision', 'message': f"  Idx {i}: Skip Split - remaining part too small ({n_messages - i} < {min_chunk_lines})"})

            # else:
                # reason = ""
                # if not (avg_drop > drop_threshold): reason += f"Drop {avg_drop:.4f} <= {drop_threshold}. "
                # if not ((i - last_split) >= min_chunk_lines): reason += f"Len {i - last_split} < {min_chunk_lines}."
                # debug_info.append({'index': i, 'type': 'split_decision', 'message': f"  Idx {i}: Skip Split - {reason}"})


    # Create chunks based on split indices
    raw_chunks = []
    split_indices.append(n_messages) # Add the end of the message list as the final split point
    # debug_info.append({'index': -1, 'type': 'final_splits', 'message': f"\nFinal Split Indices: {split_indices}"})
    for k in range(len(split_indices) - 1):
        start_idx = split_indices[k]
        end_idx = split_indices[k+1]
        chunk = messages[start_idx:end_idx]
        if chunk: # Avoid adding empty chunks
             raw_chunks.append(chunk)
             # debug_info.append({'index': start_idx, 'type': 'raw_chunk', 'message': f" Raw chunk {k}: {start_idx} to {end_idx-1} (len {len(chunk)})"})


    # Refine chunks for max length
    refined_chunks = []
    # debug_info.append({'index': -1, 'type': 'refine_info', 'message': "\nRefining chunks for max length:"})
    for i, chunk in enumerate(raw_chunks):
        if len(chunk) > max_chunk_lines:
            # debug_info.append({'index': -1, 'type': 'refine_split', 'message': f"  Chunk {i} exceeds max length ({len(chunk)} > {max_chunk_lines}), splitting:"})
            for j in range(0, len(chunk), max_chunk_lines):
                sub_chunk = chunk[j:j + max_chunk_lines]
                refined_chunks.append(sub_chunk)
                # debug_info.append({'index': -1, 'type': 'refine_sub_chunk', 'message': f"    Sub-chunk: {j} to {j + max_chunk_lines - 1} (len {len(sub_chunk)})"})
        elif len(chunk) >= min_chunk_lines: # Ensure refined chunks also meet min length (unless it was already smaller)
            refined_chunks.append(chunk)
            # debug_info.append({'index': -1, 'type': 'refine_ok', 'message': f"  Chunk {i} within limits (len {len(chunk)})"})
        # else:
             # debug_info.append({'index': -1, 'type': 'refine_skip_min', 'message': f"  Skipping chunk {i} - too short after splitting (len {len(chunk)} < {min_chunk_lines})"})


    # Final filter for min length just in case
    final_chunks = [chunk for chunk in refined_chunks if len(chunk) >= min_chunk_lines]
    # if len(final_chunks) != len(refined_chunks):
    #      debug_info.append({'index': -1, 'type': 'final_filter', 'message': f"Filtered out {len(refined_chunks) - len(final_chunks)} chunks due to min_chunk_lines after refinement."})


    print(f"Segmented into {len(final_chunks)} final chunks.")
    # Use this to write debug info if needed
    # with open("segmentation_debug.txt", 'a') as outfile:
    #      outfile.write(f"\n--- Debugging Segmentation for a file ---\n")
    #      for item in debug_info:
    #           outfile.write(f"[{item.get('index', '')} {item.get('type', '')}] {item.get('message', '')}\n")

    return final_chunks, debug_info


def embed_chunks(text_chunks, embedding_model):
    """Embeds text chunks using the provided SentenceTransformer model."""
    print(f"Embedding {len(text_chunks)} text chunks...")
    if not text_chunks:
        print("No chunks to embed.")
        return np.array([]) # Return empty numpy array

    chunk_texts = []
    for chunk in text_chunks:
        # Ensure messages within chunk are strings
        lines = [
            f"[{item['timestamp']}] {item['speaker']}: {item['message']}"
            for item in chunk
            if isinstance(item, dict) and all(key in item for key in ('timestamp', 'speaker', 'message'))
        ]
        chunk_texts.append("\n".join(lines))

    try:
        embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
        print(f"Shape of embeddings: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"Error during embedding: {e}")
        return np.array([])


def index_chunk_embeddings(index, embeddings):
    """Adds embeddings to the Faiss index."""
    if embeddings.size == 0:
        print("No embeddings to index.")
        return index # Return original index

    embeddings_np = np.array(embeddings).astype('float32')
    if embeddings_np.ndim != 2:
         print(f"Error: Embeddings have incorrect dimensions ({embeddings_np.ndim}). Expected 2.")
         return index

    if index is None:
        dimension = embeddings_np.shape[1]
        if dimension == 0:
             print("Error: Cannot create index with 0 dimension embeddings.")
             return None
        print(f"Creating new Faiss IndexFlatL2 with dimension {dimension}.")
        index = faiss.IndexFlatL2(dimension)

    if index.d != embeddings_np.shape[1]:
         print(f"Error: Index dimension ({index.d}) does not match embedding dimension ({embeddings_np.shape[1]}). Cannot add.")
         # Decide how to handle: error out, rebuild index, etc.
         # For now, just return the existing index without adding
         return index

    index.add(embeddings_np)
    print(f"Indexed {embeddings_np.shape[0]} chunk embeddings. Index total: {index.ntotal}")
    return index

def search_chunk_index(query, index, embedding_model, k=RELEVANT_CHUNK_SIZE):
    """Searches the Faiss index for relevant chunks."""
    if index is None or index.ntotal == 0:
        print("Error: Index is not available or empty for searching.")
        return ([], []), ([], []) # Return empty results

    # DEBUG print(f"Searching for relevant chunks for the query: '{query}'...")
    try:
        query_embedding = embedding_model.encode([query]).astype('float32')
        # Ensure k is not greater than the number of items in the index
        actual_k = min(k, index.ntotal)
        if actual_k == 0:
             print("Warning: Search requested but index is empty.")
             return ([], []), ([], [])
        if actual_k < k:
             print(f"Warning: Requested k={k} chunks, but only {actual_k} available in index.")

        distances, indices = index.search(query_embedding, actual_k)
        # DEBUG print(f"Found {len(indices[0])} relevant chunk indices.")
        return indices, distances
    except Exception as e:
        print(f"Error during index search: {e}")
        return ([], []), ([], []) # Return empty results on error

def generate_answer(query, relevant_chunks):
    """Generates an answer using Ollama based on relevant chunks."""
    # DEBUG print(f"Generating an answer based on {len(relevant_chunks)} relevant chunks...")
    if not relevant_chunks:
        return "Could not generate an answer because no relevant information was found."

    # Format the relevant text from chunks
    relevant_text_parts = []
    for chunk in relevant_chunks:
        chunk_part = "\n".join([f"[{item.get('timestamp', 'N/A')}] {item.get('speaker', 'Unknown')}: {item.get('message', '')}" for item in chunk if isinstance(item, dict)])
        relevant_text_parts.append(chunk_part)
    relevant_text = "\n\n---\n\n".join(relevant_text_parts) # Separate chunks clearly

    # Construct the prompt
    prompt = (
        "You are a helpful assistant analyzing a timestamped chat log between people.\n"
        "In this chat log, the person referred to as 'Me' is Scott, who is now asking you a question.\n" # Added this line
        "Use ONLY the relevant information provided below to answer the following question concisely. Do not add information not present in the log.\n\n"
        # "Provide bullet points with evidence to your answer. \n\n"
        "=== Relevant Chat Log Snippets ===\n"
        f"{relevant_text}\n\n"
        "=== End of Snippets ===\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    # print("\n--- Sending Prompt to Ollama ---")
    # print(prompt) # Optional: print the prompt for debugging
    # print("--- End Prompt ---")

    try:
        process = subprocess.Popen(
            ['ollama', 'run', 'mistral'], # Ensure 'mistral' model is available in Ollama
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8' # Explicitly set encoding
        )
        stdout, stderr = process.communicate(input=prompt, timeout=90) # Increased timeout

        # --- Refined Ollama Output Handling ---
        if process.returncode != 0:
             print(f"Error: Ollama process exited with code {process.returncode}")
             print("STDERR:")
             print(stderr)
             return f"Error generating answer (Ollama process error {process.returncode})."

        if stdout and stdout.strip():
            answer = stdout.strip()
            # Basic cleanup (sometimes Ollama includes the prompt or extra phrases)
            if "Answer:" in answer:
                 answer = answer.split("Answer:")[-1].strip()
            return answer
        else:
             # Check stderr more carefully for non-error messages (like download progress)
             if stderr and not any(indicator in stderr for indicator in ["pulling", "downloading", "verifying", "writing", "using", "success"]):
                 print("Error running Ollama (stderr indicates potential issue):")
                 print(stderr)
                 return "Error generating answer (Ollama stderr)."
             else:
                 print("Warning: Ollama returned empty stdout, but stderr seems okay. Could be an issue with the prompt or model.")
                 print("STDERR (for info):")
                 print(stderr)
                 return "Could not generate an answer (Ollama returned empty response)."
        # --- End Refined Handling ---

    except subprocess.TimeoutExpired:
        print("Error: Ollama took too long to respond.")
        process.kill()
        stdout, stderr = process.communicate() # Get any remaining output
        print("STDOUT (timeout):", stdout)
        print("STDERR (timeout):", stderr)
        return "Error: Ollama timeout."
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please ensure Ollama is installed and in your system's PATH.")
        return "Error: Ollama command not found."
    except Exception as e:
        print(f"An unexpected error occurred while running Ollama: {e}")
        print("STDERR (exception):", stderr if 'stderr' in locals() else "N/A")
        return f"Error generating answer (Exception: {e})."


def process_batch(file_paths, embedding_model):
    """Processes a batch of files: loads, chunks, embeds."""
    batch_chunks = []
    batch_embeddings_list = []
    processed_file_count = 0

    for file_path in file_paths:
        print(f"Processing file: {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    if not isinstance(data, list):
                         print(f"Warning: Data in {os.path.basename(file_path)} is not a list. Skipping.")
                         continue
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {os.path.basename(file_path)}. Skipping.")
                    continue

                if data:
                    # Chunk the messages
                    chunks, debug_info = segment_by_similarity_drop_sorted(data, embedding_model)
                    if chunks:
                        batch_chunks.extend(chunks)
                        processed_file_count += 1
                        # Optionally write chunks/debug to file (as before)
                        # with open("chunks.txt", 'a') as outfile: ...
                    else:
                         print(f"No valid chunks generated from {os.path.basename(file_path)}")
                else:
                    print(f"No messages found in {os.path.basename(file_path)}")
        except FileNotFoundError:
             print(f"Error: File not found during batch processing: {file_path}. Skipping.")
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    if batch_chunks:
        print(f"Generated {len(batch_chunks)} chunks from {processed_file_count} files in this batch.")
        # Embed the chunks from this batch
        batch_embeddings = embed_chunks(batch_chunks, embedding_model)
        if batch_embeddings.size > 0:
            return batch_chunks, batch_embeddings
        else:
            print("Warning: Embedding failed for batch chunks.")
            return [], np.array([]) # Return empty if embedding fails
    else:
        print("No chunks generated in this batch.")
        return [], np.array([]) # Return empty lists/arrays

# --- Main Execution ---

def main():
    print("Starting RAG process...\n")
    print(f"Input Path: {INPUT_PATH}")
    print(f"Cache Directory: {CACHE_DIR}")

    # --- Load Model ---
    print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=CACHE_DIR) # Specify cache folder
    except Exception as e:
        print(f"Fatal Error: Could not load SentenceTransformer model: {e}")
        print("Please ensure the model name is correct and dependencies are installed.")
        return # Exit if model cannot be loaded
    print("Model loaded.")

    # --- Load Caches ---
    index = load_cached_index()
    all_chunks = load_cached_chunks()
    cached_file_state = load_cached_file_state()
    current_file_state = get_file_state(INPUT_PATH)

    # --- Determine Files to Process ---
    files_to_process = []
    files_to_remove_from_index = [] # Keep track of files removed or changed significantly

    # Check for new/modified files
    if not current_file_state and not cached_file_state:
         print("Error: Input path seems invalid or empty, and no cache exists.")
         # Decide how to proceed - maybe exit?
    elif not current_file_state and cached_file_state:
         print("Warning: Input path seems invalid or empty, but cache exists. Using cached data only.")
         # Consider all cached files as 'removed' if they aren't in the (empty) current state
         files_to_remove_from_index = list(cached_file_state.keys())
    else: # current_file_state is populated
        all_current_files = set(current_file_state.keys())
        all_cached_files = set(cached_file_state.keys())

        new_files = all_current_files - all_cached_files
        removed_files = all_cached_files - all_current_files
        potentially_modified_files = all_current_files.intersection(all_cached_files)

        for filename in new_files:
            filepath = os.path.join(INPUT_PATH, filename)
            if os.path.exists(filepath): # Check if file still exists
                 files_to_process.append(filepath)
                 print(f"Found new file: {filename}")
            else:
                 print(f"Warning: New file {filename} detected in state but not found on disk. Ignoring.")


        for filename in removed_files:
            print(f"Detected removed file: {filename}")
            files_to_remove_from_index.append(filename) # Mark for potential removal from index (complex)

        for filename in potentially_modified_files:
            filepath = os.path.join(INPUT_PATH, filename)
            if not os.path.exists(filepath):
                 print(f"Warning: File {filename} in cache/state but not found on disk. Marking as removed.")
                 files_to_remove_from_index.append(filename)
            elif current_file_state[filename] != cached_file_state[filename]:
                print(f"Detected modified file: {filename}")
                files_to_process.append(filepath)
                files_to_remove_from_index.append(filename) # Mark old version for removal
            # else:
            #     print(f"File {filename} unchanged. Skipping.")

    # --- Rebuild Index/Chunks if necessary ---
    # Simple Strategy: If *any* file changed/added/removed, rebuild everything.
    # More Complex: Selective index removal/update (Faiss doesn't easily support removal by metadata, requires ID mapping)
    needs_rebuild = bool(files_to_process or files_to_remove_from_index)

    if needs_rebuild:
        print("\nChanges detected. Rebuilding index and chunk data...")
        all_chunks = [] # Reset chunks
        if index is not None:
             index.reset() # Clear the existing index object
             print("Cleared existing Faiss index.")
        index = None # Ensure a new index is created if needed

        # Process ALL current files (rebuild strategy)
        files_for_rebuild = []
        if os.path.isdir(INPUT_PATH):
             files_for_rebuild = [os.path.join(INPUT_PATH, f) for f in current_file_state.keys() if f.endswith(".json")]
        elif os.path.isfile(INPUT_PATH) and INPUT_PATH.endswith(".json"):
             files_for_rebuild = [INPUT_PATH]

        if not files_for_rebuild:
             print("No valid files found to rebuild index.")
        else:
            print(f"Processing {len(files_for_rebuild)} files for rebuild in batches of {BATCH_SIZE}...")
            for i in range(0, len(files_for_rebuild), BATCH_SIZE):
                batch_paths = files_for_rebuild[i:i + BATCH_SIZE]
                # Process batch returns chunks and embeddings for that batch
                batch_chunks, batch_embeddings = process_batch(batch_paths, embedding_model)

                if batch_chunks and batch_embeddings.size > 0:
                    # Add embeddings to index immediately
                    index = index_chunk_embeddings(index, batch_embeddings)
                    # Append chunks to the main list
                    all_chunks.extend(batch_chunks)
                else:
                    print(f"Warning: Batch starting with {os.path.basename(batch_paths[0])} produced no valid chunks or embeddings.")

        # --- Save Cache After Rebuild ---
        if index:
            save_cached_index(index)
        else:
             print("Warning: Index is empty after rebuild, not saving index cache.")
             if os.path.exists(INDEX_CACHE_FILE): os.remove(INDEX_CACHE_FILE) # Clean up old file

        save_cached_chunks(all_chunks) # Save the newly built chunks
        save_cached_file_state(current_file_state) # Update file state cache

    else:
        print("\nNo new or changed files detected. Using cached index and chunks.")
        # Sanity check: if index exists but chunks are missing, warn or try reload
        if index is not None and not all_chunks:
             print("Warning: Index loaded from cache, but no chunks loaded. Caches might be inconsistent.")
             print("Attempting to reload chunks from cache...")
             all_chunks = load_cached_chunks()
             if not all_chunks:
                  print("Failed to load chunks. Index may be unusable without corresponding data.")
                  # Optionally clear index here if chunks are essential
                  # index = None
                  # if os.path.exists(INDEX_CACHE_FILE): os.remove(INDEX_CACHE_FILE)

    # --- Query Loop ---
    print("\nInitialization complete.")
    while True:
        print("-" * 20)
        query = input("Ask a question about your texts (or type 'exit' to quit): ")
        query = query.strip()

        if not query:
            continue
        if query.lower() == 'exit':
            print("Exiting the program...")
            break

        if index is None or index.ntotal == 0:
            print("Error: No search index is available or the index is empty.")
            print("Please check the input files and run the script again.")
            continue # Skip to next loop iteration

        if not all_chunks:
             print("Error: No chunk data is loaded, cannot retrieve context for the answer.")
             print("This might happen if processing failed or caches are inconsistent.")
             # Maybe suggest rebuilding here
             continue

        # --- Perform Search and Answer ---
        try:
            relevant_indices, distances = search_chunk_index(query, index, embedding_model, RELEVANT_CHUNK_SIZE)

            if not isinstance(relevant_indices, np.ndarray) or relevant_indices.shape[0] == 0 or relevant_indices[0].size == 0:
                print("Could not find any relevant chunks for your query.")
                continue

            # Retrieve the actual chunks using the indices
            # Ensure index is within bounds of all_chunks list
            retrieved_chunks = []
            valid_indices = [idx for idx in relevant_indices[0] if idx < len(all_chunks)]
            if len(valid_indices) < len(relevant_indices[0]):
                 print(f"Warning: Some retrieved indices ({len(relevant_indices[0]) - len(valid_indices)}) were out of bounds for the loaded chunks list (length {len(all_chunks)}).")

            retrieved_chunks = [all_chunks[i] for i in valid_indices]

            if retrieved_chunks:
                print("\nRelevant Chunks Preview (first ~5 words of each message):")
                for i, chunk in enumerate(retrieved_chunks):
                     if chunk:
                         # Use the corrected print statement
                         preview_parts = [" ".join(item['message'].split()[:5]) for item in chunk if isinstance(item, dict) and 'message' in item]
                         print(f"- Chunk {valid_indices[i]}: {' ... '.join(preview_parts)}")
                     else:
                         print(f"- Chunk {valid_indices[i]}: [Empty Chunk Data]")

                answer = generate_answer(query, retrieved_chunks)
                print("\nAnswer:")
                print(answer)
            else:
                print("Could not retrieve chunk data for the found indices. Cache might be inconsistent.")

        except Exception as e:
             print(f"\nAn error occurred during query processing: {e}")
             import traceback
             traceback.print_exc() # Print detailed traceback for debugging

if __name__ == "__main__":
    main()