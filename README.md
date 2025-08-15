# Master Thesis ACIT 2025 Oslomet Data Science

## Towards Improving Information Retrieval through Cache-Augmented Generation: A Case Study in Norwegian Healthcare

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


## Parameters nor_kvcahe.py

- `--kvcache` *(string)* — e.g. `"file"`.
- `--dataset` *(string)* —  `"squad-train"`.
- `--similarity` *(string)* — `"bertscore"`.
- `--modelname` *(string)* — e.g. `"norallm/normistral-7b-warm-instruct"` or `"norallm/normistral-11b-warm"`.
- `--maxKnowledge` *(int)* — number of documents to use from the dataset. See **Note** below.
- `--maxParagraph` *(int)* — default `100`.
- `--maxQuestion` *(int)* — max number of questions. See **Note** below.
- `--randomSeed` *(int)* — random seed.
- `--output` *(string)* — output file path.
- `--usePrompt` *(flag)* — include this flag if **not** using CAG knowledge-cache acceleration.

> [!NOTE]
> - For `--maxKnowledge` and `--maxQuestion`, set an integer to limit, or omit to use all available.
> - Quotes are optional unless the value contains spaces.


##Example -- kvcache.py

```bash
python ./nor_kvcache.py --kvcache file --dataset "squad-train" --similarity bertscore \
    --maxKnowledge 5 --maxParagraph 100 --maxQuestion 1000  \
    --modelname "norallm/normistral-7b-warm-instruct" --randomSeed 0 \
    --output "./result_kvcache.txt"

```

##Parameter Usage -- rag.py
- `--index` *(string)* —  `"bm25"`.
- `--dataset` *(string)* — `"squad-train"`.
- `--similarity` *(string)* — `"bertscore"`.
- `--maxKnowledge` *(int)* — number of documents to use from the dataset. See **Note** below.
- `--maxParagraph` *(int)* — default `100`.
- `--maxQuestion` *(int)* — max number of questions. See **Note** below.
- `--topk` *(int)* — top-K results to keep from the retrieval step.
- `--modelname` *(string)* — e.g. `"norallm/normistral-7b-warm-instruct"` or `"norallm/normistral-11b-warm"`..
- `--randomSeed` *(int)* — random seed.
- `--output` *(string)* — output file path.

> [!NOTE]
> - For `--maxKnowledge` and `--maxQuestion`, set an integer to limit, or omit to use all available.
> - Quotes around values are optional unless the value contains spaces.
##Example -- rag.py

```bash
python ./rag.py --index "bm25" --dataset "squad-train" --similarity bertscore \
    --maxKnowledge 80 --maxParagraph 100 --maxQuestion 80 --topk 3 \
    --modelname "norallm/normistral-7b-warm-instruct" --randomSeed  0 \
    --output  "./rag_results.txt"
```
