import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import VectorStoreIndex, Document
from transformers.cache_utils import DynamicCache
import cag.dataset as cagds
import cag.similarity as cagsim
import argparse
import os
from transformers import BitsAndBytesConfig
import logging
from config import ConfigName, set_config
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

global model_name, model, tokenizer

# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

from time import time
from llama_index.core import Settings


def getOpenAIRetriever(documents: list[Document], similarity_top_k: int = 1):
    """OpenAI RAG model"""
    import openai
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found")
    openai.api_key = OPENAI_API_KEY

    from llama_index.embeddings.openai import OpenAIEmbedding
    Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", api_key=OPENAI_API_KEY,
                                           title="openai-embedding")

    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    OpenAI_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()
    logger.info(f"OpenAI retriever prepared in {t2 - t1:.2f} seconds.")
    return OpenAI_retriever, t2 - t1


def getGeminiRetriever(documents: list[Document], similarity_top_k: int = 1):
    """Gemini Embedding RAG model"""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found")
    from llama_index.embeddings.gemini import GeminiEmbedding
    model_name = "models/embedding-001"
    Settings.embed_model = GeminiEmbedding(model_name=model_name, api_key=GOOGLE_API_KEY, title="gemini-embedding")

    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    Gemini_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()
    logger.info(f"Gemini retriever prepared in {t2 - t1:.2f} seconds.")
    return Gemini_retriever, t2 - t1


def getBM25Retriever(documents: list[Document], similarity_top_k: int = 1):
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.retrievers.bm25 import BM25Retriever
    import Stemmer

    # Use same chunking as CAG for fair comparison
    splitter = SentenceSplitter(chunk_size=512)

    t1 = time()
    nodes = splitter.get_nodes_from_documents(documents)

    try:
        stemmer = Stemmer.Stemmer("norwegian")
        language = "norwegian"
        logger.info("Using Norwegian stemmer for BM25")
    except:
        stemmer = Stemmer.Stemmer("english")
        language = "english"
        logger.warning("Norwegian stemmer not available, using English stemmer")

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=stemmer,
        language=language,
    )
    t2 = time()
    bm25_retriever.persist("./bm25_retriever")

    return bm25_retriever, t2 - t1


def getJinaRetriever(documents: list[Document], similarity_top_k: int = 1):
    """Jina Embedding model"""
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY not found")
    try:
        from llama_index.embeddings.jinaai import JinaEmbedding
        model_name = "jina-embeddings-v3"
        Settings.embed_model = JinaEmbedding(
            api_key=JINA_API_KEY,
            model=model_name,
            task="retrieval.passage",
        )

        t1 = time()
        index = VectorStoreIndex.from_documents(documents)
        Jina_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        t2 = time()
        logger.info(f"Jina retriever prepared in {t2 - t1:.2f} seconds.")
        return Jina_retriever, t2 - t1
    except ImportError:
        logger.error("Failed to import JinaEmbedding. Please install jinaai package.")
        raise
    except Exception as e:
        logger.error(f"Error creating Jina retriever: {str(e)}")
        raise


def generate_improved(model, tokenizer, input_ids: torch.Tensor, past_key_values=None, max_new_tokens: int = 200) -> torch.Tensor:
    """
    Improved generation function with better handling for Norwegian models.
    """
    embed_device = model.model.embed_tokens.weight.device
    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)

    output_ids = input_ids.clone()
    next_token = input_ids

    # Handle eos_token_id properly
    eos_token_ids = model.config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    
    # Add pad_token_id as stop token if available
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in eos_token_ids:
        eos_token_ids.append(tokenizer.pad_token_id)

    generated_tokens = 0
    repetition_window = 10  # Check last 10 tokens for repetition
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature and top_p for better generation quality
            temperature = 0.7
            top_p = 0.9
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top_p filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if past_key_values is not None:
                past_key_values = outputs.past_key_values

            output_ids = torch.cat([output_ids, next_token], dim=1)
            generated_tokens += 1

            # Check for EOS tokens
            if any(token.item() in eos_token_ids for token in next_token.view(-1)):
                break
                
            # Check for repetitive generation
            if generated_tokens >= repetition_window:
                recent_tokens = output_ids[0, -repetition_window:].tolist()
                if len(set(recent_tokens)) < 3:  # Too few unique tokens in recent window
                    logger.warning("Repetitive generation detected, stopping early")
                    break

    return output_ids[:, origin_ids.shape[-1]:]


def clean_generated_text(text: str) -> str:
    """
    Clean the generated text to remove unwanted artifacts.
    """
    # Remove repeated patterns
    text = re.sub(r'(.{1,10})\1{3,}', r'\1', text)
    
    # Remove non-Norwegian characters (keep Norwegian letters)
    text = re.sub(r'[^\w\s\.,!?æøåÆØÅ-]', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def rag_test(args: argparse.Namespace):
    # Use same instruction as CAG
    answer_instruction = "Du er en norsk sykehusassistent på et sykehus. Du svarer kort og presist på spørsmål basert på dokumentasjon fra ehåndbok.no eller opplastet kontekst."

    text_list, dataset = cagds.get(args.dataset, max_knowledge=args.maxKnowledge,
                                   max_paragraph=args.maxParagraph, max_questions=args.maxQuestion)

    # Remove duplicates in context (same as CAG)
    text_list = list(dict.fromkeys(text_list))

    # Document indexing for the rag retriever
    documents = [Document(text=t) for t in text_list]

    retriever = None
    prepare_time = 0.0
    if args.index == "gemini":
        retriever, prepare_time = getGeminiRetriever(documents, similarity_top_k=args.topk)
    if args.index == "openai":
        retriever, prepare_time = getOpenAIRetriever(documents, similarity_top_k=args.topk)
        logger.info(f"Testing {args.index.upper()} retriever with {len(documents)} documents.")
    if args.index == "bm25":
        retriever, prepare_time = getBM25Retriever(documents, similarity_top_k=args.topk)
    if args.index == "jina":
        retriever, prepare_time = getJinaRetriever(documents, similarity_top_k=args.topk)
        logger.info(f"Testing {args.index.upper()} retriever with {len(documents)} documents.")

    if retriever is None:
        raise ValueError("No retriever, `--index` not set")

    print(f"Retriever {args.index.upper()} prepared in {prepare_time} seconds")
    with open(args.output, "a") as f:
        f.write(f"Retriever {args.index.upper()} prepared in {prepare_time} seconds\n")

    results = {
        "retrieve_time": [],
        "generate_time": [],
        "similarity": [],
        "prompts": [],
        "responses": []
    }

    dataset = list(dataset)
    max_questions = min(len(dataset), args.maxQuestion) if args.maxQuestion != None else len(dataset)

    for id, (question, ground_truth) in enumerate(dataset[:max_questions]):

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        retrieve_t1 = time()
        nodes = retriever.retrieve(question)
        retrieve_t2 = time()

        knowledge = "\n\n\n".join([node.text for node in nodes])

        # Simplified prompt format that should work better
        prompt = f"""<s>[INST] {answer_instruction}

Kontekst:
{knowledge}

Spørsmål: {question}

Svar kort og presist på norsk: [/INST]"""

        print("---------------------------PROMPT----------------------------")
        print(prompt)

        # Generate Response using improved method
        generate_t1 = time()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # Use improved generation function
        output = generate_improved(model, tokenizer, input_ids, past_key_values=None, max_new_tokens=150)
        generate_t2 = time()

        generated_ids = output[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean the generated text
        generated_text = clean_generated_text(generated_text)

        # Extract the answer only (remove any prompt artifacts)
        if "Svar:" in generated_text:
            generated_text = generated_text.split("Svar:")[1].strip()
        
        # Additional cleaning for Norwegian context
        generated_text = generated_text.strip()

        print("Q: ", question)
        print("A: ", generated_text)

        # Check for repetitive output (improved detection)
        words = generated_text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.4:  # Less than 40% unique words
                print("⚠️ Repetitive output detected!")
                # Try to salvage by taking first few unique words
                seen = set()
                clean_words = []
                for word in words:
                    if word not in seen and len(clean_words) < 10:
                        seen.add(word)
                        clean_words.append(word)
                generated_text = " ".join(clean_words)

        # Evaluate bert-score similarity (same as CAG)
        similarity = cagsim.bert(generated_text, ground_truth) if generated_text.strip() else 0.0

        print(f"[{id}]: Semantic Similarity: {round(similarity, 5)},",
              f"retrieve time: {retrieve_t2 - retrieve_t1},",
              f"generate time: {generate_t2 - generate_t1}")

        with open(args.output, "a") as f:
            f.write(
                f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t retrieve time: {retrieve_t2 - retrieve_t1},\t generate time: {generate_t2 - generate_t1}\n")

        results["prompts"].append(prompt)
        results["responses"].append(generated_text)
        results["retrieve_time"].append(retrieve_t2 - retrieve_t1)
        results["generate_time"].append(generate_t2 - generate_t1)
        results["similarity"].append(similarity)

        with open(args.output, "a") as f:
            f.write(f"[{id}]: [Cumulative]: "
                    + f"Semantic Similarity: {round(sum(results['similarity']) / len(results['similarity']), 5)},"
                    + f"\t retrieve time: {sum(results['retrieve_time']) / len(results['retrieve_time'])},"
                    + f"\t generate time: {sum(results['generate_time']) / len(results['generate_time'])}\n")

    avg_similarity = sum(results["similarity"]) / len(results["similarity"])
    avg_retrieve_time = sum(results["retrieve_time"]) / len(results["retrieve_time"])
    avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])

    print()
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {avg_similarity}")
    print(f"retrieve time: {avg_retrieve_time},\t generate time: {avg_generate_time}")
    print()

    with open(args.output, "a") as f:
        f.write("\n")
        f.write(f"Result for {args.output}\n")
        f.write(f"Prepare time: {prepare_time}\n")
        f.write(f"Average Semantic Similarity: {avg_similarity}\n")
        f.write(f"retrieve time: {avg_retrieve_time},\t generate time: {avg_generate_time}\n")


# Define quantization configuration (same as CAG)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


def load_quantized_model(model_name, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )
    return tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG test with specified parameters.")
    parser.add_argument('--modelname', required=False, default="norallm/normistral-7b-warm-instruct", type=str,
                        help='Model name to use')
    parser.add_argument('--quantized', required=False, default=False, type=bool, help='Quantized model')
    parser.add_argument('--index', choices=['gemini', 'openai', 'bm25', 'jina'], required=True,
                        help='Index to use (gemini, openai, bm25, jina)')
    parser.add_argument('--similarity', choices=['bertscore'], required=True,
                        help='Similarity metric to use (bertscore)')
    parser.add_argument('--output', required=True, type=str, help='Output file to save the results')
    parser.add_argument('--maxQuestion', required=False, default=None, type=int,
                        help='Maximum number of questions to test')
    parser.add_argument('--maxKnowledge', required=False, default=None, type=int,
                        help='Maximum number of knowledge items to use')
    parser.add_argument('--maxParagraph', required=False, default=None, type=int,
                        help='Maximum number of paragraph to use')
    parser.add_argument('--topk', required=False, default=3, type=int, help='Top K retrievals to use')
    parser.add_argument('--dataset', required=True, help='Dataset to use',
                        choices=['kis', 'kis_sample',
                                 'squad-dev', 'squad-train',
                                 'hotpotqa-dev', 'hotpotqa-train', 'hotpotqa-test',
                                 'norallm', 'cag_ruter', 'llama_inst'])
    parser.add_argument('--randomSeed', required=False, default=None, type=int, help='Random seed to use')

    args = parser.parse_args()

    print("maxKnowledge", args.maxKnowledge, "maxParagraph", args.maxParagraph, "maxQuestion", args.maxQuestion,
          "randomSeed", args.randomSeed)

    model_name = args.modelname
    if args.randomSeed != None:
        set_config(ConfigName.RAND_SEED, args.randomSeed)

    if args.quantized:
        tokenizer, model = load_quantized_model(model_name=model_name, hf_token=HF_TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN
        )

    # Ensure tokenizer has proper padding token (same as CAG)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def unique_path(path, i=0):
        if os.path.exists(path):
            return unique_path(path + "_" + str(i), i + 1)
        return path

    if os.path.exists(args.output):
        args.output = unique_path(args.output)

    rag_test(args)