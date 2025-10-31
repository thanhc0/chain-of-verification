from datasets import load_dataset
import pandas as pd

def load_truthfulqa_dataset(sample_size=3, output_csv=True):
    """
    Load the TruthfulQA (generation) dataset and optionally export the first n samples to CSV.

    Args:
        sample_size (int): number of samples to select
        output_csv (bool): whether to save the subset to a CSV file

    Returns:
        pd.DataFrame: subset of the dataset as a DataFrame
    """
    print(f"⏳ Loading TruthfulQA (generation) dataset (sample={sample_size})...")
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset("truthful_qa", "generation")

        # Select subset from validation split
        subset = dataset["validation"].select(range(sample_size))

        # Convert to pandas DataFrame
        df = pd.DataFrame(subset)

        print(f"✅ Successfully loaded {len(df)} samples.")

        # Save to CSV
        if output_csv:
            output_filename = f"TruthfulQA_{sample_size}.csv"
            df.to_csv(output_filename, index=False, encoding="utf-8")
            print(f"💾 Saved to file: {output_filename}")

        # Show preview
        print("\n--- Sample preview ---")
        print(df.head(3)[["question", "best_answer"]])

        return df

    except Exception as e:
        print(f"❌ Error loading TruthfulQA: {e}")
        return None


# Example usage
if __name__ == "__main__":
    load_truthfulqa_dataset(sample_size=200)
