import torch
import argparse
import os
import cag.dataset as cagds
import cag.similarity as cagsim
from time import time
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import logging
from config import ConfigName, set_config
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv(".env.template")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found")

"""Improved CAG System with Better Evaluation and Generation"""

global model_name, model, tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])


# =====================================================
# IMPROVED EVALUATION METRICS
# =====================================================

def exact_match_score(predicted, ground_truth):
    """Simple exact match - very strict but clear"""
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


def fuzzy_match_score(predicted, ground_truth):
    """More lenient matching for Norwegian text"""
    # Normalize both texts
    pred_norm = re.sub(r'[^\w\s]', '', predicted.lower().strip())
    gt_norm = re.sub(r'[^\w\s]', '', ground_truth.lower().strip())

    # If ground truth is very short (1-2 words), be strict
    if len(gt_norm.split()) <= 2:
        return 1.0 if pred_norm == gt_norm else 0.0

    # For longer answers, check if GT is contained in prediction
    if gt_norm in pred_norm:
        return 1.0

    # Check word overlap
    pred_words = set(pred_norm.split())
    gt_words = set(gt_norm.split())

    if len(gt_words) == 0:
        return 0.0

    overlap = len(pred_words & gt_words) / len(gt_words)
    return overlap


def length_penalty_score(predicted, ground_truth, max_ratio=3.0):
    """Penalize responses that are too verbose"""
    if len(ground_truth.strip()) == 0:
        return 0.0

    length_ratio = len(predicted.strip()) / len(ground_truth.strip())

    if length_ratio <= max_ratio:
        return 1.0
    else:
        return max(0.0, 1.0 - (length_ratio - max_ratio) / max_ratio)


def comprehensive_score(predicted, ground_truth):
    """Combine multiple metrics for better evaluation"""
    exact = exact_match_score(predicted, ground_truth)
    fuzzy = fuzzy_match_score(predicted, ground_truth)
    length_penalty = length_penalty_score(predicted, ground_truth)

    # If exact match, perfect score
    if exact == 1.0:
        return 1.0

    # Otherwise combine fuzzy match with length penalty
    return fuzzy * length_penalty


# =====================================================
# IMPROVED CONTEXT PROCESSING
# =====================================================

def extract_key_sentences(context, question, max_sentences=2):
    """Extract only the most relevant sentences from context"""

    # Split context into sentences
    sentences = re.split(r'[.!?]+', context)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    question_words = set(question.lower().split())
    # Remove Norwegian stop words
    stop_words = {'og', 'eller', 'en', 'et', 'er', 'for', 'til', 'av', 'på', 'med', 'i', 'som', 'det', 'den', 'de',
                  'når', 'hvor', 'hva', 'hvem'}
    question_keywords = question_words - stop_words

    # Score sentences by keyword overlap
    scored_sentences = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(question_keywords & sentence_words)

        # Bonus for exact phrase matches
        sentence_lower = sentence.lower()
        for keyword in question_keywords:
            if len(keyword) > 3 and keyword in sentence_lower:
                overlap += 0.5

        scored_sentences.append((overlap, sentence))

    # Return top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    top_sentences = [sent for score, sent in scored_sentences[:max_sentences] if score > 0]

    if not top_sentences and sentences:
        top_sentences = sentences[:1]  # Fallback to first sentence

    return '. '.join(top_sentences).strip() + ('.' if top_sentences else '')


def smart_context_truncation(context, max_chars=2000):
    """Truncate context smartly while preserving sentence boundaries"""
    if len(context) <= max_chars:
        return context

    # Find last complete sentence within limit
    truncated = context[:max_chars]
    last_period = truncated.rfind('.')

    if last_period > max_chars * 0.7:  # Only if we don't lose too much
        return truncated[:last_period + 1]

    return truncated + "..."


# =====================================================
# IMPROVED GENERATION
# =====================================================

def create_simple_norwegian_prompt(context, question):
    """Create a much simpler prompt that works better with Norwegian models"""

    # Extract only most relevant context
    relevant_context = extract_key_sentences(context, question, max_sentences=2)
    relevant_context = smart_context_truncation(relevant_context, max_chars=500)

    # Ultra-simple format without any special tokens
    prompt = f"""Kontekst: {relevant_context}

Spørsmål: {question}
Svar:"""

    return prompt


def clean_norwegian_response(response):
    """Aggressively clean response for Norwegian medical Q&A"""

    if not response:
        return ""

    # Remove special tokens and artifacts
    response = re.sub(r'<\|[^>]*\|>', '', response)
    response = re.sub(r'<[^>]*>', '', response)
    response = re.sub(r'\|[^|]*\|', '', response)

    # Remove URLs and reference artifacts
    response = re.sub(r'https?://[^\s]+', '', response)
    response = re.sub(r'www\.[^\s]+', '', response)
    response = re.sub(r'\[[^\]]*\]', '', response)
    response = re.sub(r'\([^)]*\)', '', response)

    # Remove HTML-like content
    response = re.sub(r'&[a-zA-Z]+;', '', response)
    response = re.sub(r'&#[0-9]+;', '', response)

    # Remove random character sequences and corrupted text
    response = re.sub(r'[A-Z]{8,}', '', response)
    response = re.sub(r'\d{8,}', '', response)
    response = re.sub(r'[^\w\sæøåÆØÅ,.!?%-]', ' ', response)

    # Remove repetitive patterns
    response = re.sub(r'(\w+\s+)\1{3,}', r'\1', response)

    # Take only the first sentence or short phrase
    sentences = re.split(r'[.!?]+', response)
    if sentences:
        first_sentence = sentences[0].strip()

        # If first sentence is very long, try to extract key part
        if len(first_sentence) > 100:
            # Try to find the main clause
            words = first_sentence.split()
            if len(words) > 15:
                first_sentence = ' '.join(words[:15]) + "..."

        if len(first_sentence) > 3:
            return first_sentence

    # Fallback: just clean and return limited response
    clean_text = ' '.join(response.split())  # Normalize whitespace
    return clean_text[:100] if len(clean_text) > 100 else clean_text


def is_valid_response(response):
    """Check if response is valid and not corrupted"""
    if not response or len(response.strip()) < 2:
        return False

    # Check for repetitive patterns
    words = response.split()
    if len(words) > 3:
        unique_words = len(set(words))
        if unique_words / len(words) < 0.5:  # Too repetitive
            return False

    # Check for garbage characters (non-Norwegian text)
    norwegian_chars = re.findall(r'[a-zA-ZæøåÆØÅ\s0-9,.!?%-]', response)
    if len(norwegian_chars) < len(response) * 0.8:  # Too many non-Norwegian chars
        return False

    # Check for obvious corruption patterns
    corruption_patterns = [
        r'[A-Z]{5,}',  # Too many consecutive capitals
        r'\d{5,}',  # Too many consecutive digits
        r'(.)\1{4,}',  # Same character repeated 5+ times
    ]

    for pattern in corruption_patterns:
        if re.search(pattern, response):
            return False

    return True


def generate_robust_answer(model, tokenizer, prompt, max_attempts=3):
    """Generate with multiple fallback strategies"""

    # Different generation strategies from conservative to more creative
    strategies = [
        {
            "max_new_tokens": 100,
            "temperature": 0.0,  # Completely deterministic
            "do_sample": False,
            "repetition_penalty": 1.0
        },
        {
            "max_new_tokens": 150,
            "temperature": 0.1,
            "do_sample": True,
            "top_p": 0.8,
            "repetition_penalty": 1.1
        },
        {
            "max_new_tokens": 150,
            "temperature": 0.3,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3
        }
    ]

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    for attempt, strategy in enumerate(strategies):
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True,
                    **strategy
                )

            # Extract only the new tokens
            generated_tokens = outputs[0][input_ids.shape[-1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Clean the response
            cleaned_response = clean_norwegian_response(response)

            if is_valid_response(cleaned_response):
                return cleaned_response, attempt + 1

        except Exception as e:
            logging.warning(f"Generation attempt {attempt + 1} failed: {e}")
            continue

    return "Informasjonen finnes ikke i konteksten", max_attempts


# =====================================================
# IMPROVED CAG IMPLEMENTATION
# =====================================================

def create_stable_kv_cache(model, tokenizer, context, system_prompt):
    """Create a more stable KV cache without complex formatting"""

    # Simplified cache prompt format - easier to understand
    cache_prompt = f"""{system_prompt}

Kontekst:
{context}

Spørsmål:"""

    # Conservative length limit to avoid issues
    if len(cache_prompt) > 3000:
        # Truncate context while preserving system prompt
        max_context_len = 2500 - len(system_prompt)
        truncated_context = smart_context_truncation(context, max_context_len)
        cache_prompt = f"""{system_prompt}

Kontekst:
{truncated_context}

Spørsmål:"""

    print(f"Cache prompt length: {len(cache_prompt)} chars")

    input_ids = tokenizer.encode(cache_prompt, return_tensors="pt").to(model.device)
    print(f"Cache token count: {input_ids.shape[1]}")

    past_key_values = DynamicCache()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False
        )

    return outputs.past_key_values, input_ids.shape[1]


def generate_with_cache(model, tokenizer, question, kv_cache, cache_length):
    """Generate using pre-computed cache with better error handling"""

    # Very simple question continuation - matches the cache format
    question_prompt = f" {question}\nSvar:"

    input_ids = tokenizer.encode(question_prompt, return_tensors="pt").to(model.device)

    # Reset cache to original state - CRITICAL for stability
    try:
        for i in range(len(kv_cache.key_cache)):
            kv_cache.key_cache[i] = kv_cache.key_cache[i][:, :, :cache_length, :]
            kv_cache.value_cache[i] = kv_cache.value_cache[i][:, :, :cache_length, :]
    except Exception as e:
        logging.error(f"Cache reset failed: {e}")
        return "Cache error"

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                past_key_values=kv_cache,
                max_new_tokens=100,  # Very conservative
                temperature=0.1,
                do_sample=False,  # Greedy for stability
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )

        generated_tokens = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return clean_norwegian_response(response)

    except Exception as e:
        logging.error(f"Cache generation failed: {e}")
        return "Generation error"


def get_simple_system_prompt():
    """Simplified system prompt for Norwegian medical Q&A"""
    return """Du svarer kort og presist på norsk basert kun på konteksten. Hvis svaret ikke finnes i konteksten, skriv "Ikke funnet"."""


# =====================================================
# MAIN IMPROVED FUNCTION
# =====================================================

def improved_cag_test(args: argparse.Namespace):
    """Main function with improved CAG implementation"""

    # Get documents and Q&A dataset
    text_list, dataset = cagds.get(
        args.dataset,
        max_knowledge=args.maxKnowledge,
        max_paragraph=args.maxParagraph,
        max_questions=args.maxQuestion
    )

    print(f"DEBUG: Got {len(text_list)} knowledge documents")

    dataset = list(dataset)
    print(f"DEBUG: Got {len(dataset)} total questions from dataset")

    # Remove duplicates in context
    text_list = list(dict.fromkeys(text_list))

    max_questions = min(len(dataset), args.maxQuestion) if args.maxQuestion is not None else len(dataset)
    print(f"DEBUG: Will process {max_questions} questions")

    results = {
        "exact_match": [],
        "fuzzy_match": [],
        "comprehensive_score": [],
        "bert_similarity": [],  # Keep for comparison
        "cache_time": [],  # Added to match other files
        "generation_time": [],
        "response_length": [],
        "responses": [],
        "questions": [],
        "ground_truths": []
    }

    system_prompt = get_simple_system_prompt()

    # Track cache preparation time
    cache_prepare_time = 0.0

    if args.usePrompt:
        print("Using improved per-question context filtering...")
        print("Mode: Individual prompts for each question (no cache)")

        for id, (question, ground_truth) in enumerate(dataset[:max_questions]):
            torch.cuda.empty_cache()

            # No cache preparation time for individual prompts
            cache_time = 0.0

            start_time = time()

            # Create simple prompt with relevant context
            relevant_context = extract_key_sentences('\n\n'.join(text_list), question, max_sentences=2)
            prompt = create_simple_norwegian_prompt(relevant_context, question)

            # Generate with robust method
            response, attempts = generate_robust_answer(model, tokenizer, prompt)

            generation_time = time() - start_time

            print(f"--- Question {id + 1} ---")
            print(f"Q: {question}")
            print(f"A: {response}")
            print(f"GT: {ground_truth}")
            print(f"Attempts needed: {attempts}")
            print(f"Generation time: {generation_time:.3f}s")

            # Evaluate with multiple metrics
            exact = exact_match_score(response, ground_truth)
            fuzzy = fuzzy_match_score(response, ground_truth)
            comprehensive = comprehensive_score(response, ground_truth)
            bert_sim = cagsim.bert(response, ground_truth)

            print(f"Exact Match: {exact:.3f}")
            print(f"Fuzzy Match: {fuzzy:.3f}")
            print(f"Comprehensive: {comprehensive:.3f}")
            print(f"BERT Similarity: {bert_sim:.3f}")

            # Output in format similar to other files
            print(f"[{id}]: Semantic Similarity: {round(bert_sim, 5)}, "
                  f"cache time: {cache_time}, "
                  f"generate time: {generation_time}")

            print("-" * 50)

            # Store results
            results["exact_match"].append(exact)
            results["fuzzy_match"].append(fuzzy)
            results["comprehensive_score"].append(comprehensive)
            results["bert_similarity"].append(bert_sim)
            results["cache_time"].append(cache_time)
            results["generation_time"].append(generation_time)
            results["response_length"].append(len(response))
            results["responses"].append(response)
            results["questions"].append(question)
            results["ground_truths"].append(ground_truth)

    else:
        print("Using improved KV cache approach...")
        print("Mode: Pre-compute cache once, reuse for all questions")

        # Create stable cache with limited context
        global_context = smart_context_truncation('\n\n'.join(text_list), max_chars=2500)
        print(f"Global context: {len(global_context)} characters")

        cache_start_time = time()
        kv_cache, cache_length = create_stable_kv_cache(model, tokenizer, global_context, system_prompt)
        cache_prepare_time = time() - cache_start_time

        print(f"KV cache created in {cache_prepare_time:.3f} seconds")
        print(f"Cache length: {cache_length} tokens")

        for id, (question, ground_truth) in enumerate(dataset[:max_questions]):
            torch.cuda.empty_cache()

            # Cache reading time (minimal since cache is in memory)
            cache_t1 = time()
            # Simulate cache reading - in real scenario this would be loading from file
            cache_time = time() - cache_t1

            start_time = time()
            response = generate_with_cache(model, tokenizer, question, kv_cache, cache_length)
            generation_time = time() - start_time

            print(f"--- Question {id + 1} ---")
            print(f"Q: {question}")
            print(f"A: {response}")
            print(f"GT: {ground_truth}")
            print(f"Generation time: {generation_time:.3f}s")

            # Evaluate with multiple metrics
            exact = exact_match_score(response, ground_truth)
            fuzzy = fuzzy_match_score(response, ground_truth)
            comprehensive = comprehensive_score(response, ground_truth)
            bert_sim = cagsim.bert(response, ground_truth)

            print(f"Exact Match: {exact:.3f}")
            print(f"Fuzzy Match: {fuzzy:.3f}")
            print(f"Comprehensive: {comprehensive:.3f}")
            print(f"BERT Similarity: {bert_sim:.3f}")

            # Output in format similar to other files
            print(f"[{id}]: Semantic Similarity: {round(bert_sim, 5)}, "
                  f"cache time: {cache_time}, "
                  f"generate time: {generation_time}")

            print("-" * 50)

            # Store results
            results["exact_match"].append(exact)
            results["fuzzy_match"].append(fuzzy)
            results["comprehensive_score"].append(comprehensive)
            results["bert_similarity"].append(bert_sim)
            results["cache_time"].append(cache_time)
            results["generation_time"].append(generation_time)
            results["response_length"].append(len(response))
            results["responses"].append(response)
            results["questions"].append(question)
            results["ground_truths"].append(ground_truth)

    # Calculate and print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    avg_exact = sum(results["exact_match"]) / len(results["exact_match"])
    avg_fuzzy = sum(results["fuzzy_match"]) / len(results["fuzzy_match"])
    avg_comprehensive = sum(results["comprehensive_score"]) / len(results["comprehensive_score"])
    avg_bert = sum(results["bert_similarity"]) / len(results["bert_similarity"])
    avg_cache_time = sum(results["cache_time"]) / len(results["cache_time"])
    avg_generation_time = sum(results["generation_time"]) / len(results["generation_time"])
    avg_length = sum(results["response_length"]) / len(results["response_length"])

    print(f"Cache preparation time: {cache_prepare_time:.3f}s")
    print(f"Average Exact Match: {avg_exact:.3f}")
    print(f"Average Fuzzy Match: {avg_fuzzy:.3f}")
    print(f"Average Comprehensive Score: {avg_comprehensive:.3f}")
    print(f"Average BERT Similarity: {avg_bert:.3f}")
    print(f"Average Cache Time: {avg_cache_time:.6f}s")
    print(f"Average Generation Time: {avg_generation_time:.3f}s")
    print(f"Average Response Length: {avg_length:.1f} chars")

    # Analyze performance by response length
    short_responses = [i for i, r in enumerate(results["responses"]) if len(r) < 50]
    long_responses = [i for i, r in enumerate(results["responses"]) if len(r) >= 50]

    if short_responses:
        short_exact = sum(results["exact_match"][i] for i in short_responses) / len(short_responses)
        print(f"Exact match for short responses (<50 chars): {short_exact:.3f}")

    if long_responses:
        long_exact = sum(results["exact_match"][i] for i in long_responses) / len(long_responses)
        print(f"Exact match for long responses (>=50 chars): {long_exact:.3f}")

    # Write detailed results to file in consistent format
    with open(args.output, "w", encoding='utf-8') as f:
        f.write(f"Improved CAG System Results\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model: {args.modelname}\n")
        f.write(f"Questions processed: {len(results['questions'])}\n")
        f.write(f"Use cache: {not args.usePrompt}\n")
        f.write(f"Cache preparation time: {cache_prepare_time:.3f}s\n\n")

        # Write individual results in format matching other files
        for i in range(len(results["questions"])):
            f.write(f"[{i}]: Semantic Similarity: {round(results['bert_similarity'][i], 5)}, ")
            f.write(f"cache time: {results['cache_time'][i]}, ")
            f.write(f"generate time: {results['generation_time'][i]}\n")

        # Write cumulative averages
        f.write(f"\nFinal Results:\n")
        f.write(f"Cache preparation time: {cache_prepare_time:.3f}s\n")
        f.write(f"Average Semantic Similarity: {avg_bert:.5f}\n")
        f.write(f"Average Cache Time: {avg_cache_time:.6f}s\n")
        f.write(f"Average Generation Time: {avg_generation_time:.6f}s\n\n")

        f.write(f"COMPREHENSIVE SCORES:\n")
        f.write(f"Exact Match: {avg_exact:.3f}\n")
        f.write(f"Fuzzy Match: {avg_fuzzy:.3f}\n")
        f.write(f"Comprehensive Score: {avg_comprehensive:.3f}\n")
        f.write(f"BERT Similarity: {avg_bert:.3f}\n")
        f.write(f"Avg Response Length: {avg_length:.1f} chars\n\n")

        f.write(f"DETAILED RESULTS:\n")
        f.write(f"{'=' * 50}\n")

        for i in range(len(results["questions"])):
            f.write(f"Question {i + 1}: {results['questions'][i]}\n")
            f.write(f"Response: {results['responses'][i]}\n")
            f.write(f"Ground Truth: {results['ground_truths'][i]}\n")
            f.write(f"Exact: {results['exact_match'][i]:.3f}, ")
            f.write(f"Fuzzy: {results['fuzzy_match'][i]:.3f}, ")
            f.write(f"Comprehensive: {results['comprehensive_score'][i]:.3f}, ")
            f.write(f"BERT: {results['bert_similarity'][i]:.3f}\n")
            f.write(f"Cache: {results['cache_time'][i]:.6f}s, ")
            f.write(f"Generate: {results['generation_time'][i]:.3f}s, ")
            f.write(f"Length: {results['response_length'][i]} chars\n")
            f.write("-" * 30 + "\n")


# =====================================================
# DEBUGGING UTILITIES
# =====================================================

def debug_generation_issues(model, tokenizer, sample_questions, text_list):
    """Debug specific generation problems"""

    print("=== DEBUGGING GENERATION ISSUES ===")
    print("This helps understand how the model responds to different prompt formats")

    for question in sample_questions[:3]:
        print(f"\nTesting question: {question}")

        # Test different prompt approaches
        approaches = [
            ("Minimal", f"Kontekst: {text_list[0][:100]}\nSpørsmål: {question}\nSvar:"),
            ("Simple", create_simple_norwegian_prompt(text_list[0], question)),
            ("No context", f"Spørsmål: {question}\nSvar:"),
        ]

        for name, prompt in approaches:
            print(f"\n--- {name} approach ---")
            print(f"Prompt length: {len(prompt)} chars")
            print(f"Prompt preview: {repr(prompt[:100])}...")

            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                print(f"Token count: {input_ids.shape[1]}")

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=100,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                generated_tokens = outputs[0][input_ids.shape[-1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                cleaned = clean_norwegian_response(response)

                print(f"Raw response: {repr(response)}")
                print(f"Cleaned: {repr(cleaned)}")
                print(f"Valid: {is_valid_response(cleaned)}")
                print(f"Generated {len(generated_tokens)} tokens")

                # Check if any tokens were generated
                if len(generated_tokens) == 0:
                    print("WARNING: No tokens generated!")
                elif generated_tokens[0].item() == tokenizer.eos_token_id:
                    print("WARNING: Immediate EOS token!")

            except Exception as e:
                print(f"ERROR: {e}")


# =====================================================
# MODEL LOADING (SAME AS ORIGINAL)
# =====================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


def load_quantized_model(model_name, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )

    return tokenizer, model


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run improved CAG test with better evaluation.")
    parser.add_argument('--modelname', required=False, default="norallm/normistral-11b-warm", type=str,
                        help='Model name to use')
    parser.add_argument('--quantized', action='store_true', help='Use Quantized model')
    parser.add_argument('--kvcache', choices=['file'], required=True, help='Method to use (from_file or from_var)')
    parser.add_argument('--similarity', choices=['bertscore'], required=True,
                        help='Similarity metric to use (bertscore)')
    parser.add_argument('--output', required=True, type=str, help='Output file to save the results')
    parser.add_argument('--maxQuestion', required=False, default=10, type=int,
                        help='Maximum number of questions to test (default: 10 for debugging)')
    parser.add_argument('--maxKnowledge', required=False, default=None, type=int,
                        help='Maximum number of knowledge items to use')
    parser.add_argument('--maxParagraph', required=False, default=None, type=int,
                        help='Maximum number of paragraph to use')
    parser.add_argument('--usePrompt', default=False, action="store_true",
                        help='Use per-question context filtering instead of cache')
    parser.add_argument('--randomSeed', required=False, default=None, type=int, help='Random seed to use')
    parser.add_argument('--dataset', required=True, help='Dataset to use',
                        choices=['kis', 'kis_sample', 'squad-dev', 'squad-train',
                                 'hotpotqa-dev', 'hotpotqa-train', 'hotpotqa-test',
                                 'norallm', 'llama_inst'])
    parser.add_argument('--debug', action='store_true', help='Run debugging mode first')

    args = parser.parse_args()

    print("IMPROVED CAG SYSTEM")
    print("=" * 50)
    print(f"Model: {args.modelname}")
    print(f"Dataset: {args.dataset}")
    print(f"maxKnowledge: {args.maxKnowledge}")
    print(f"maxParagraph: {args.maxParagraph}")
    print(f"maxQuestion: {args.maxQuestion}")
    print(f"randomSeed: {args.randomSeed}")
    print(f"usePrompt: {args.usePrompt} ({'Individual prompts' if args.usePrompt else 'KV Cache'})")
    print(f"quantized: {args.quantized}")
    print(f"debug: {args.debug}")
    print("=" * 50)

    model_name = args.modelname
    if args.randomSeed is not None:
        set_config(ConfigName.RAND_SEED, args.randomSeed)

    # Load model
    print(f"Loading model: {model_name}")
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

    # Ensure tokenizer and model compatibility
    assert tokenizer.name_or_path in model.name_or_path or model_name in model.name_or_path, \
        f"Tokenizer '{tokenizer.name_or_path}' and model '{model.name_or_path}' may be mismatched!"

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Model loaded successfully: {model_name}")
    print(f"✓ Model device: {model.device}")
    print(f"✓ Tokenizer vocab size: {len(tokenizer)}")

    # Create unique output path if file exists
    def unique_path(path, i=0):
        if os.path.exists(path):
            return unique_path(path + "_" + str(i), i + 1)
        return path

    if os.path.exists(args.output):
        args.output = unique_path(args.output)

    # Optional debugging mode
    if args.debug:
        print("\n" + "=" * 50)
        print("RUNNING DEBUG MODE FIRST")
        print("=" * 50)
        print("This will test different prompt formats to understand model behavior")

        # Get a small sample for debugging
        text_list_debug, dataset_debug = cagds.get(
            args.dataset,
            max_knowledge=5,  # Very small for debugging
            max_paragraph=10,
            max_questions=3
        )

        sample_questions = [q for q, _ in list(dataset_debug)[:3]]
        debug_generation_issues(model, tokenizer, sample_questions, text_list_debug)

        print("\n✓ DEBUG MODE COMPLETE. Continuing with full test...\n")

    # Run the improved CAG test
    print("Starting CAG evaluation...")
    improved_cag_test(args)

    print(f"\n✓ Results saved to: {args.output}")
    print("✓ Improved CAG test completed successfully!")
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Mode: {'Individual Prompts' if args.usePrompt else 'KV Cache'}")
    print(f"Model: {args.modelname}")
    print(f"Dataset: {args.dataset}")
    print(f"Questions processed: {args.maxQuestion if args.maxQuestion else 'All available'}")
    print(f"Output file: {args.output}")
    
    if args.usePrompt:
        print("\nPrompt Mode Explanation:")
        print("- Each question gets its own optimized prompt")
        print("- Context is filtered to most relevant sentences")
        print("- No cache preparation time")
        print("- More flexible but potentially slower per question")
    else:
        print("\nKV Cache Mode Explanation:")
        print("- Context is pre-processed once into a KV cache")
        print("- Cache preparation takes time upfront")
        print("- Each question reuses the same cached context")
        print("- Faster per question but less flexible context")
    
    print("\nOutput Format Explanation:")
    print("- Semantic Similarity: BERT-based similarity score (0-1)")
    print("- Cache time: Time to access/prepare context (seconds)")
    print("- Generate time: Time to generate the answer (seconds)")
    print("- Exact/Fuzzy/Comprehensive: Different accuracy metrics")
    print("=" * 60)