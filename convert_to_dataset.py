import os
import json
import argparse
from datasets import Dataset, DatasetDict, load_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--synthetic_data_path",
        required=True,
        help="Synthetic data path to load generated examples",
    )
    parser.add_argument(
        "--output_data_path", required=True, help="Dataset name to save as"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset = load_dataset("gsm8k", "main")
    test_dataset = dataset["test"]

    generated_examples = list(
        map(json.loads, open(f"synthetic_data/{args.synthetic_data_path}.jsonl"))
    )

    train = {
        "question": [],
        "answer": [],
        "generated_code_string": [],
    }
    test = {
        "question": [],
        "answer": [],
        "generated_code_string": [],
    }

    for generated_example in generated_examples:
        if (
            generated_example["score"] == 1
            and "Counter" in generated_example["program"]
            and "assert " in generated_example["program"]
        ):
            train["question"].append(generated_example["question"])
            train["answer"].append(generated_example["answer"])
            train["generated_code_string"].append(generated_example["program"])

    for example in test_dataset:
        test["question"].append(example["question"])
        test["answer"].append(example["answer"])
        test["generated_code_string"].append("")

    train_dataset = Dataset.from_dict(train)
    test_dataset = Dataset.from_dict(test)

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    print(dataset)

    dataset.save_to_disk(f"data/{args.output_data_path}")


if __name__ == "__main__":
    main()
