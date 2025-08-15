import json
import random
import pandas as pd
from typing import Iterator
from config import ConfigName, get_config

rand_seed = None

def _parse_squad_data(raw):
    dataset = {"ki_text": [], "qas": []}

    for k_id, data in enumerate(raw["data"]):
        article = []
        for p_id, para in enumerate(data["paragraphs"]):
            article.append(para["context"])
            for qa in para["qas"]:
                ques = qa["question"]
                answers = [ans["text"] for ans in qa["answers"]]
                dataset["qas"].append(
                    {
                        "title": data["title"],
                        "paragraph_index": tuple((k_id, p_id)),
                        "question": ques,
                        "answers": answers,
                    }
                )
        dataset["ki_text"].append(
            {"id": k_id, "title": data["title"], "paragraphs": article}
        )

    return dataset


def squad(
    filepath: str,
    max_knowledge: int | None = None,
    max_paragraph: int | None = None,
    max_questions: int | None = None,
) -> tuple[list[str], Iterator[tuple[str, str]]]:
    """
    @param filepath: path to the dataset's JSON file
    @param max_knowledge: maximum number of docs in dataset
    @param max_paragraph:
    @param max_questions:
    @return: knowledge list, question & answer pair list
    """
    # Open and read the JSON file
    with open(filepath, "r") as file:
        data = json.load(file)
    # Parse the SQuAD data
    parsed_data = _parse_squad_data(data)

    print(
        "max_knowledge",
        max_knowledge,
        "max_paragraph",
        max_paragraph,
        "max_questions",
        max_questions,
    )

    # Set the limit Maximum Articles, use all Articles if max_knowledge is None or greater than the number of Articles
    max_knowledge = (
        max_knowledge
        if max_knowledge is not None and max_knowledge < len(parsed_data["ki_text"])
        else len(parsed_data["ki_text"])
    )
    max_paragraph = max_paragraph if max_knowledge == 1 else None

    # Shuffle the Articles and Questions
    if rand_seed is not None:
        print("rand_seed: ", rand_seed)
        random.seed(rand_seed)
        random.shuffle(parsed_data["ki_text"])
        random.shuffle(parsed_data["qas"])

    k_ids = [i["id"] for i in parsed_data["ki_text"][:max_knowledge]]

    text_list = []
    # Get the knowledge Articles for at most max_knowledge, or all Articles if max_knowledge is None
    for article in parsed_data["ki_text"][:max_knowledge]:
        max_para = (
            max_paragraph
            if max_paragraph is not None and max_paragraph < len(article["paragraphs"])
            else len(article["paragraphs"])
        )
        text_list.append(article["title"])
        text_list.append("\n".join(article["paragraphs"][0:max_para]))

    # Check if the knowledge id of qas is less than the max_knowledge
    questions = [
        qa["question"]
        for qa in parsed_data["qas"]
        if qa["paragraph_index"][0] in k_ids
        and (max_paragraph is None or qa["paragraph_index"][1] < max_paragraph)
    ]
    answers = [
        qa["answers"][0]
        for qa in parsed_data["qas"]
        if qa["paragraph_index"][0] in k_ids
        and (max_paragraph is None or qa["paragraph_index"][1] < max_paragraph)
    ]

    dataset = zip(questions, answers)

    return text_list, dataset


def hotpotqa(
    filepath: str, max_knowledge: int | None = None
) -> tuple[list[str], Iterator[tuple[str, str]]]:
    """
    @param filepath: path to the dataset's JSON file
    @param max_knowledge:
    @return: knowledge list, question & answer pair list
    """
    # Open and read the JSON
    with open(filepath, "r") as file:
        data = json.load(file)

    if rand_seed is not None:
        print("rand_seed: ", rand_seed)
        random.seed(rand_seed)
        random.shuffle(data)

    questions = [qa["question"] for qa in data]
    answers = [qa["answer"] for qa in data]
    dataset = zip(questions, answers)

    if max_knowledge is None:
        max_knowledge = len(data)
    else:
        max_knowledge = min(max_knowledge, len(data))

    text_list = []
    for _, qa in enumerate(data[:max_knowledge]):
        context = qa["context"]
        context = [c[0] + ": \n" + "".join(c[1]) for c in context]
        article = "\n\n".join(context)

        text_list.append(article)

    return text_list, dataset


def kis(filepath: str) -> tuple[list[str], Iterator[tuple[str, str]]]:
    """
    @param filepath: path to the dataset's JSON file
    @return: knowledge list, question & answer pair list
    """
    df = pd.read_csv(filepath)
    dataset = zip(df["sample_question"], df["sample_ground_truth"])
    text_list = df["ki_text"].to_list()

    return text_list, dataset


def get(
    dataset: str,
    max_knowledge: int | None = None,
    max_paragraph: int | None = None,
    max_questions: int | None = None,
) -> tuple[list[str], Iterator[tuple[str, str]]]:
    global rand_seed
    rand_seed = get_config(ConfigName.RAND_SEED)
    match dataset:
        case "kis_sample":
            path = "./datasets/rag_sample_qas_from_kis.csv"
            return kis(path)
        case "kis":
            path = "./datasets/synthetic_knowledge_items.csv"
            return kis(path)
        case "squad-dev":
            path = "./datasets/squad/dev-v1.1.json"
            return squad(path, max_knowledge, max_paragraph, max_questions)
        case "squad-train":
            path = "./datasets/squad/heart_100_high.json" #train-v1.1.json
            return squad(path, max_knowledge, max_paragraph, max_questions)
        case "hotpotqa-dev":
            path = "./datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
            return hotpotqa(path, max_knowledge)
        case "hotpotqa-test":
            path = "./datasets/hotpotqa/hotpot_test_fullwiki_v1.json"
            return hotpotqa(path, max_knowledge)
        case "hotpotqa-train":
            path = "./datasets/hotpotqa/hotpot_train_v1.1.json"
            return hotpotqa(path, max_knowledge)
        
        case "norallm":
            path = "./datasets/norwegian_ruter_large_new_format.json"
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            if rand_seed is not None:
                random.seed(rand_seed)
                random.shuffle(data)

            # Limit by number of questions if requested
            if max_questions is not None:
                data = data[:max_questions]

            text_list = [item["input"] for item in data]  # For building KV cache
            qa_pairs = [(item["input"], item["output"]) for item in data]

            print(f"[norallm] Loaded {len(qa_pairs)} QA pairs.")
            return text_list, qa_pairs

            #text_list = [item["input"] for item in data]
            #qa_pairs = [(item["input"], item["output"]) for item in data]

            # Apply max_question limit if provided
            #if max_questions is not None:
                #text_list = text_list[:max_questions]
                #qa_pairs = qa_pairs[:max_questions]

            #print(f"Loaded {len(text_list)} entries from Norwegian dataset.")
            #return text_list, qa_pairs
        
        case "cag_ruter":
            path = "./datasets/cag_ready_data.json"
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            if rand_seed is not None:
                random.seed(rand_seed)
                random.shuffle(data)

            if max_questions is not None:
                data = data[:max_questions]

            text_list = []
            qa_pairs = []

            for item in data:
                full_input = item["input"]
                answer = item["output"]

                try:
                    # Split input into context and question
                    context_part, question_part = full_input.split("Spørsmål:", 1)
                    context = context_part.replace("Kontekst:", "").strip()
                    question = question_part.strip()
                except ValueError:
                    # Fallback if structure is not as expected
                    context = ""
                    question = full_input.strip()

                # Build prompt in CAG system format
                prompt = f"""<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        Du er en assistent på et sykehus. Du svarer kort og presist på spørsmål basert på dokumentasjonen som er tilgjengelig – enten fra opplastet kontekst eller fra retningslinjer på ehåndbok.no.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Her er konteksten:
        ----------------------------------------
        {context}
        ----------------------------------------
        Svar på spørsmålet med et veldig kort svar.
        Spørsmål: {question}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        Svar:"""

                text_list.append(prompt)
                qa_pairs.append((question, answer))

            print(f"[cag_ruter] Loaded {len(qa_pairs)} QA pairs.")
            return text_list, qa_pairs
        
        case "llama_inst":
            path = "./datasets/dataset_try.json"
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            if rand_seed is not None:
                random.seed(rand_seed)
                random.shuffle(data)

            if max_questions is not None:
                data = data[:max_questions]

            text_list = []
            qa_pairs = []

            for item in data:
                prompt = item["input"]
                answer = item["output"]

                # You can also extract the question from the prompt if needed, but here we treat full prompt as input
                text_list.append(prompt)
                qa_pairs.append((prompt, answer))

            print(f"[llama_inst] Loaded {len(qa_pairs)} QA pairs.")
            return text_list, qa_pairs

        case _:
            return [], zip([], [])
        
        #added for the norwegian dataset
        
