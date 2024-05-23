import os
import json
import tqdm
import copy
import argparse
from interface import ProgramChatInterface, ClassificationChatInterface
from prompts import (
    UNIT_CONSISTENCY_FEW_SHOT_PROMPT,
    UNIT_CONSISTENCY_QUESTION_PROMPT,
    UNIT_CONSISTENCY_SYSTEM_PROMPT,
    CLASSIFICATION_SYSTEM_MESSAGE,
    CLASSIFICATION_FEW_SHOT_PROMPT,
)
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        default="program",
        help="Task of generation. program or classification.",
    )
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", help="Dataset for generation"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4-1106-preview", help="Model for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature value used for prompting",
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top p value used for promting"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="Max tokens used for prompting"
    )
    parser.add_argument(
        "--end_problem", type=int, default=-1, help="Last problem for generation"
    )
    parser.add_argument("--openai_key", type=str, required=True, help="Openai Key")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    OUTPUT_PATH = f"synthetic_data/{args.dataset}_{args.task}"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if args.task == "program":
        dataset = load_dataset("gsm8k", "main")["train"]

        interface = ProgramChatInterface(
            model=args.model,
            system_message=UNIT_CONSISTENCY_SYSTEM_PROMPT,
            openai_key=args.openai_key,
        )
        path = f"{OUTPUT_PATH}.jsonl"
        with open(path, "+a") as f:
            lines = open(path).readlines()
            start_problem = len(lines)
            scores = [x["score"] for x in map(json.loads, lines)]

            if args.end_problem == -1:
                end_problem = len(dataset)
            else:
                end_problem = args.end_problem

            print("*" * 40)
            print(f"Model: {args.model}")
            print(f"Number of examples: {len(dataset)}")
            print(f"starting problem: {start_problem}")
            print(f"ending problem: {end_problem}")
            print(f"output file: {path}")
            print("*" * 40)

            progress_bar = tqdm.tqdm(
                dataset.select(list(range(start_problem, end_problem))),
                initial=start_problem,
                total=end_problem,
            )
            for example in progress_bar:

                question = example["question"]
                result = copy.copy(example)
                prompt = (
                    UNIT_CONSISTENCY_FEW_SHOT_PROMPT
                    + UNIT_CONSISTENCY_QUESTION_PROMPT.format(question=question)
                )

                for idx in range(1):
                    while True:
                        try:
                            program, code, ans = interface.run(
                                prompt=prompt,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=args.max_tokens,
                            )
                            break
                        except Exception as e:
                            print(f"Interface Error: {e}")
                            continue

                    try:
                        score = 1 if abs(ans - example["answer"]) < 1e-3 else 0
                    except Exception as e:
                        print(f"Wrong Answer")
                        ans = ""
                        score = 0

                    if score == 1:
                        print(f"iteration {idx} pass")
                        break
                    else:
                        print(f"iteration {idx} fail")

                result["predicted_answer"] = ans
                result["score"] = score
                result["prompt"] = prompt
                result["program"] = code
                result["generated_program"] = program
                f.write(json.dumps(result) + "\n")

                scores.append(score)
                f.flush()

        print(f"Accuracy - {sum(scores) / len(scores)}")

    if args.task == "classification":
        if args.dataset == "gsm8k":
            dataset = load_dataset("gsm8k", "main")

            interface = ClassificationChatInterface(
                model=args.model,
                system_message=CLASSIFICATION_SYSTEM_MESSAGE,
                openai_key=args.openai_key,
            )

            for split in dataset.keys():
                dataset_split = dataset[split]

                path = f"{OUTPUT_PATH}_{split}.jsonl"
                with open(path, "a+") as f:
                    lines = open(path).readlines()
                    start_problem = len(lines)

                    if args.end_problem == -1:
                        end_problem = len(dataset_split)
                    else:
                        end_problem = args.end_problem

                    print("*" * 40)
                    print(f"Model: {args.model}")
                    print(f"Number of examples: {len(dataset_split)}")
                    print(f"starting problem: {start_problem}")
                    print(f"ending problem: {end_problem}")
                    print(f"output file: {path}")
                    print("*" * 40)

                    progress_bar = tqdm.tqdm(
                        dataset_split.select(list(range(start_problem, end_problem))),
                        initial=start_problem,
                        total=end_problem,
                    )
                    for example in progress_bar:
                        question = example["question"]
                        result = copy.copy(example)

                        prompt = CLASSIFICATION_FEW_SHOT_PROMPT.format(
                            question=question
                        )
                        explanation, label = interface.run(
                            prompt=prompt,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )

                        result["explanation"] = explanation
                        result["label"] = label

                        f.write(json.dumps(result) + "\n")
                        f.flush()

        elif args.dataset == "svamp":
            dataset = load_dataset("ChilleD/SVAMP")

            interface = ClassificationChatInterface(
                model=args.model,
                system_message=CLASSIFICATION_SYSTEM_MESSAGE,
                openai_key=args.openai_key,
            )

            for split in dataset.keys():
                dataset_split = dataset[split]

                dataset_split = dataset_split.filter(
                    lambda example: example["Type"].startswith("Common-Division")
                    or example["Type"].startswith("Multiplication")
                )

                path = f"{OUTPUT_PATH}_{split}.jsonl"
                with open(path, "a+") as f:
                    lines = open(path).readlines()
                    start_problem = len(lines)

                    if args.end_problem == -1:
                        end_problem = len(dataset_split)
                    else:
                        end_problem = args.end_problem

                    print("*" * 40)
                    print(f"Model: {args.model}")
                    print(f"Number of examples: {len(dataset_split)}")
                    print(f"starting problem: {start_problem}")
                    print(f"ending problem: {end_problem}")
                    print(f"output file: {path}")
                    print("*" * 40)

                    progress_bar = tqdm.tqdm(
                        dataset_split.select(list(range(start_problem, end_problem))),
                        initial=start_problem,
                        total=end_problem,
                    )
                    for example in progress_bar:
                        question = f"{example['Body']} {example['Question']}"
                        result = copy.copy(example)
                        result["full question"] = question

                        prompt = CLASSIFICATION_FEW_SHOT_PROMPT.format(
                            question=question
                        )
                        explanation, label = interface.run(
                            prompt=prompt,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )

                        result["explanation"] = explanation
                        result["label"] = label

                        f.write(json.dumps(result) + "\n")
                        f.flush()


if __name__ == "__main__":
    main()
