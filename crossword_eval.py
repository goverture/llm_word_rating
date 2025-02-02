import os
import re
import json
import threading
from typing import Optional, Union
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# Define the Pydantic model for the expected JSON output.
class WordEvaluation(BaseModel):
    word: str = Field(..., description="The evaluated word")
    analysis: str = Field(..., description="The chain-of-thought explanation")
    rating: int = Field(..., description="The quality score (integer between 10 and 50)")

# Generate the JSON schema from the model.
GUIDED_JSON: Optional[Union[str, dict, BaseModel]] = WordEvaluation.model_json_schema()

# ------------------------------------------------------------------------------
# Import vllm and set up the offline inference model.
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Create an LLM object. Replace "facebook/opt-125m" with your desired model.
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
# Set sampling parameters; using a lower temperature for more consistent outputs.
guided_decoding_params = GuidedDecodingParams(json=GUIDED_JSON)

sampling_params = SamplingParams(
    temperature=0.2,
    top_p=0.95,
    max_tokens=1024,
    guided_decoding=guided_decoding_params
)

# ------------------------------------------------------------------------------
# Input and output filenames.
INPUT_FILE = "wordlist.txt"
OUTPUT_FILE = "results.csv"  # Each line will have: word;rating

# A lock to protect file writes.
file_lock = threading.Lock()

def load_words(filename: str):
    """Load the list of words from the given file (one per line)."""
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_processed_words(filename: str):
    """
    Load already processed words from the output file.
    This allows the process to resume if interrupted.
    """
    processed = set()
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                # Expecting each line in the format: word;rating
                parts = line.split(";")
                if parts:
                    processed.add(parts[0].strip())
    return processed

def append_result(filename: str, evaluation: WordEvaluation):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{evaluation.word};{evaluation.rating}\n")

def extract_json_data(response_text: str) -> WordEvaluation:
    """
    Extract the JSON data from the model response and validate it against the WordEvaluation model.
    """
    try:
        # Locate the first JSON block in the response.
        json_str = re.search(r"\{.*\}", response_text, re.DOTALL).group(0)
        data = json.loads(json_str)
        # Validate and parse the JSON data using the Pydantic model.
        evaluation = WordEvaluation(**data)
        return evaluation
    except Exception as e:
        raise ValueError("Unable to parse JSON data from response: " + str(e))

def generate_prompt(word: str, guided_json: Optional[Union[str, dict, BaseModel]] = None) -> str:
    """
    Generate the prompt for evaluating a given word.
    """
    json_instructions = ""
    if guided_json:
        # Convert guided_json to a JSON-formatted string.
        if isinstance(guided_json, dict):
            guided_json_str = json.dumps(guided_json, indent=2)
        elif isinstance(guided_json, BaseModel):
            guided_json_str = guided_json.json(indent=2)
        else:
            guided_json_str = guided_json
        json_instructions = (
            "\nAdditionally, please output your final result in a JSON format that strictly follows this schema:\n"
            f"{guided_json_str}\n"
            "Make sure the JSON object is valid and contains all required fields."
        )
    
    prompt = (
        "Evaluate the following word for its suitability in a crossword puzzle grid. "
        "Rate its quality on a scale from 10 to 50, where:\n"
        "  - 10 indicates a low quality (e.g. typos, unknown or random strings),\n"
        "  - 50 indicates a high quality (e.g. common, interesting words).\n\n"
        "Please provide your chain-of-thought reasoning first, then on a new line output the final result in JSON format as specified."
        + json_instructions +
        "\n\n"
        "For example:\n"
        "  - For the word 'asdfg': It appears to be a random string or typo, so it should receive a rating of 10.\n"
        "    Output: {\"word\": \"asdfg\", \"analysis\": \"It appears to be a random string or typo.\", \"rating\": 10}\n\n"
        "  - For the word 'apple': It is a common and interesting word, making it a good candidate, so it might receive a rating of 50.\n"
        "    Output: {\"word\": \"apple\", \"analysis\": \"It is a common and interesting word.\", \"rating\": 50}\n\n"
        f"Now, evaluate the word: '{word}'."
    )
    return prompt

def main():
    # Load words and determine which ones have already been processed.
    words = load_words(INPUT_FILE)
    processed_words = load_processed_words(OUTPUT_FILE)
    total_words = len(words)
    initial_count = len(processed_words)
    print(f"Total words to evaluate: {total_words}. Already processed: {initial_count}.")

    # Filter out words that have already been processed.
    words_to_process = [word for word in words if word not in processed_words]
    remaining = len(words_to_process)
    print(f"Words left to process: {remaining}.")

    completed_count = initial_count
    BATCH_SIZE = 32

    # Process words in batches.
    for i in range(0, len(words_to_process), BATCH_SIZE):
        batch_words = words_to_process[i: i + BATCH_SIZE]
        # Generate prompts for each word in the current batch.
        prompts = [generate_prompt(word, guided_json=GUIDED_JSON) for word in batch_words]

        # Perform batched inference using vllm.
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        
        # Process each output.
        for word, output in zip(batch_words, outputs):
            generated_text = output.outputs[0].text.strip()
            print(f"\nRaw model output for '{word}':\n{generated_text}\n")
            try:
                evaluation = extract_json_data(generated_text)
            except Exception as e:
                print(f"Error processing word '{word}': {e}")
                continue

            print(f"Extracted data - Word: {evaluation.word}, Rating: {evaluation.rating}\n")
            with file_lock:
                append_result(OUTPUT_FILE, evaluation)

            completed_count += 1
            percentage = (completed_count / total_words) * 100
            print(f"Progress: {completed_count}/{total_words} ({percentage:.2f}%)\n")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully.")
