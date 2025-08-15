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



[!IMPORTANT]
Create your .env from the template and add the required keys:

cp ./.env.template ./.env
