#Master Thesis ACIT 2025 Oslomet Data Science

#Towards Improving Information Retrieval
through Cache-Augmented Generation: A
Case Study in Norwegian Healthcare

We conduct a comparative evaluation between CAG and RAG using two LLM configurations—
7B and 11B parameters—tested across multiple question sets (20, 50, and 100 queries)
representing procedural, administrative, and clinical content. System performance was assessed
using BERT semantic similarity scores, fuzzy matching accuracy, generation time measurements,
and domain expert qualitative evaluation. Our results show that optimal CAG configurations
achieved a BERT similarity of 0.767 and fuzzy matching score of 0.498, performing
competitively with RAG benchmarks. Notably, the 7B parameter model consistently outperformed
the 11B variant, highlighting the role of domain-specific optimization over raw model
size in specialized settings.


## Installation

```bash
pip install -r ./requirements.txt

```


##[!IMPORTANT]
Create your .env from the template and add the required keys:

```bash

cp ./.env.template ./.env

```

##Usage
nor_rag.py is for RAG Experiment
nor_kvcache.py is for CAG Experiment


## Parameter Usage -- kvcache.py
--kvcache: "file"
--dataset: "hotpotqa-train" or "squad-train"
--similarity "bertscore"
--modelname: "meta-llama/Llama-3.1-8B-Instruct"
--maxKnowledge: "", int, select how many document in dataset, explanation in Note
--maxParagraph: 100
--maxQuestion int, max question number, explanation in Note
--randomSeed: "", int, a random seed number
--output: "", str, output filepath string
--usePrompt, add this parameter if not using CAG knowledge cache acceleration
