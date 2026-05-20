import random
import json
import os

def sample_jsonl(input_filepath, output_filepath, n_samples):
    """
    Randomly samples n_samples lines from an input jsonl file and writes them to an output file.

    Args:
        input_filepath (str): Path to the input jsonl file.
        output_filepath (str): Path to the output jsonl file.
        n_samples (int): Number of lines to sample.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        if len(lines) < n_samples:
            print(f"Warning: The input file has only {len(lines)} lines, which is less than the requested {n_samples} samples. Sampling all lines.")
            sampled_lines = lines
        else:
            sampled_lines = random.sample(lines, n_samples)

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for line in sampled_lines:
                outfile.write(line)

        print(f"Successfully sampled {len(sampled_lines)} lines from '{os.path.basename(input_filepath)}' and saved to '{os.path.basename(output_filepath)}'.")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define file paths relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'college-math-test.jsonl')
    output_file = os.path.join(script_dir, 'college-math-test-600.jsonl')
    num_samples = 600

    sample_jsonl(input_file, output_file, num_samples)
