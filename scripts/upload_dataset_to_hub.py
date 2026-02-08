import argparse
from datasets import load_dataset
from huggingface_hub import upload_file

def main():
    parser = argparse.ArgumentParser(description="Upload a dataset to the Hugging Face Hub.")
    parser.add_argument("--dataset_path", type=str, default="data/unsloth/alpaca.jsonl", help="Path to the dataset file to upload.")
    parser.add_argument("--datacard_path", type=str, default="datacard.md", help="Path to the dataset card file.")
    parser.add_argument("--repo_name", type=str, required=True, help="Name of the repository on the Hugging Face Hub (e.g., 'your-username/your-dataset-name').")
    parser.add_argument("--hf_token", type=str, default=None, help="Your Hugging Face API token (if not already logged in).")

    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset_path}")
    # Load the dataset as a single split
    dataset = load_dataset('json', data_files=args.dataset_path, split='train')

    print(f"Pushing dataset to Hugging Face Hub repository: {args.repo_name}")
    # Push the dataset to the hub
    dataset.push_to_hub(args.repo_name, token=args.hf_token)

    print(f"Uploading datacard from: {args.datacard_path}")
    # Upload the datacard as README.md
    upload_file(
        path_or_fileobj=args.datacard_path,
        path_in_repo="README.md",
        repo_id=args.repo_name,
        repo_type="dataset",
        token=args.hf_token,
    )

    print("\nDataset and datacard uploaded successfully!")
    print(f"You can now view your dataset at: https://huggingface.co/datasets/{args.repo_name}")

if __name__ == "__main__":
    main()
