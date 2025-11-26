"""Experiment framework for comparing LLM performance with different data formats."""

import json
import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from google import genai

from .csv_to_english import convert_all_to_english
from .data_loader import (
    get_all_csv_as_string,
    get_all_data_as_json,
    load_metadata,
)

logger = logging.getLogger(__name__)


class DataFormat(str, Enum):
    """The data formats we're testing."""
    RAW_CSV = "raw_csv"
    CSV_WITH_METADATA = "csv_with_metadata"
    ENGLISH_SENTENCES = "english_sentences"
    JSON = "json"
    JSON_WITH_METADATA = "json_with_metadata"


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    question: str
    data_format: DataFormat
    answer: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str
    system_prompt: str = ""
    user_prompt: str = ""
    raw_response: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Results comparing all three formats for one question."""
    question: str
    expected_answer: str | None
    results: dict[DataFormat, ExperimentResult]
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "results": {
                fmt.value: {
                    "answer": res.answer,
                    "latency_ms": res.latency_ms,
                    "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens,
                }
                for fmt, res in self.results.items()
            }
        }


# Research questions designed to test different types of reasoning
# Difficulty: easy (simple lookup), medium (filtering/aggregation), hard (multi-join/calculation)
RESEARCH_QUESTIONS = [
    # === EASY - Simple lookups and counts ===
    {
        "question": "What is the most expensive product in the catalog?",
        "expected": "The Smart Watch Fitness (P010) at $199.99",
        "type": "lookup",
        "difficulty": "easy"
    },
    {
        "question": "Which supplier has the best reliability rating?",
        "expected": "TechWorld Distribution (SUP001) with a rating of 4.8",
        "type": "lookup",
        "difficulty": "easy"
    },
    {
        "question": "How many customers are in the VIP segment?",
        "expected": "2 customers (Emily Nakamura and Robert Kim)",
        "type": "count",
        "difficulty": "easy"
    },
    {
        "question": "What is the cheapest product?",
        "expected": "Organic Green Tea (50 bags) at $12.99",
        "type": "lookup",
        "difficulty": "easy"
    },
    {
        "question": "How many products are in the catalog?",
        "expected": "12 products",
        "type": "count",
        "difficulty": "easy"
    },
    # === MEDIUM - Filtering and simple aggregation ===
    {
        "question": "How many orders are still pending or processing?",
        "expected": "4 orders (2 Pending, 2 Processing)",
        "type": "filtering",
        "difficulty": "medium"
    },
    {
        "question": "What is the total revenue from all orders?",
        "expected": "The total revenue is approximately $2,544.54",
        "type": "aggregation",
        "difficulty": "medium"
    },
    {
        "question": "How many products are in the Electronics category?",
        "expected": "4 products",
        "type": "filtering",
        "difficulty": "medium"
    },
    {
        "question": "What payment method is used most frequently?",
        "expected": "Credit Card (8 orders)",
        "type": "aggregation",
        "difficulty": "medium"
    },
    {
        "question": "Which customer has placed the most orders?",
        "expected": "Multiple customers tied at 2 orders each",
        "type": "aggregation",
        "difficulty": "medium"
    },
    # === HARD - Multi-table joins and complex calculations ===
    {
        "question": "Which product category has generated the most revenue?",
        "expected": "Electronics (~$1,109.88)",
        "type": "aggregation_with_join",
        "difficulty": "hard"
    },
    {
        "question": "What is the average profit margin percentage across all products?",
        "expected": "~119% average margin",
        "type": "calculation",
        "difficulty": "hard"
    },
    {
        "question": "What products did customer Emily Nakamura purchase across all her orders?",
        "expected": "Smart Watch Fitness, Wireless Mouse Ergonomic, Wireless Bluetooth Headphones",
        "type": "multi_join",
        "difficulty": "hard"
    },
    {
        "question": "Which supplier's products have generated the most total revenue?",
        "expected": "TechWorld Distribution (SUP001)",
        "type": "multi_join",
        "difficulty": "hard"
    },
    {
        "question": "What is the average order value for VIP customers vs Standard customers?",
        "expected": "VIP customers have higher average order values",
        "type": "complex_aggregation",
        "difficulty": "hard"
    },
    # === HARD - Inference and pattern recognition ===
    {
        "question": "What customer attributes are most indicative of higher spending?",
        "expected": "VIP segment customers and those in certain geographic regions tend to spend more",
        "type": "inference",
        "difficulty": "hard"
    },
    {
        "question": "Which product characteristics correlate with higher sales volume?",
        "expected": "Lower price points and certain categories like Electronics tend to have higher volume",
        "type": "inference",
        "difficulty": "hard"
    },
    {
        "question": "Are there any suppliers whose products appear to be underperforming?",
        "expected": "Analysis of supplier product sales vs catalog presence",
        "type": "inference",
        "difficulty": "hard"
    },
    {
        "question": "What patterns do you notice in shipping method preferences across customer segments?",
        "expected": "VIP customers may prefer express shipping; patterns vary by segment",
        "type": "inference",
        "difficulty": "hard"
    },
    {
        "question": "Based on the data, which product categories might benefit from expanding inventory?",
        "expected": "Categories with high sales velocity and good margins",
        "type": "inference",
        "difficulty": "hard"
    },
    {
        "question": "What insights can you draw about customer loyalty from repeat purchase patterns?",
        "expected": "Analysis of customers with multiple orders and their characteristics",
        "type": "inference",
        "difficulty": "hard"
    },
]


def _prepare_data_prompt(data_format: DataFormat) -> str:
    """Prepare the data section of the prompt based on format."""
    if data_format == DataFormat.RAW_CSV:
        return f"""Here is the data in CSV format:

{get_all_csv_as_string()}"""
    
    elif data_format == DataFormat.CSV_WITH_METADATA:
        metadata = load_metadata()
        metadata_str = json.dumps(metadata, indent=2)
        return f"""Here is the database schema metadata:

{metadata_str}

Here is the data in CSV format:

{get_all_csv_as_string()}"""
    
    elif data_format == DataFormat.ENGLISH_SENTENCES:
        english_data = convert_all_to_english()
        return f"""Here is the e-commerce data described in natural language:

{english_data}"""
    
    elif data_format == DataFormat.JSON:
        json_data = get_all_data_as_json()
        return f"""Here is the data in JSON format:

{json_data}"""
    
    elif data_format == DataFormat.JSON_WITH_METADATA:
        metadata = load_metadata()
        metadata_str = json.dumps(metadata, indent=2)
        json_data = get_all_data_as_json()
        return f"""Here is the database schema metadata:

{metadata_str}

Here is the data in JSON format:

{json_data}"""
    
    else:
        raise ValueError(f"Unknown data format: {data_format}")


def run_single_experiment(
    question: str,
    data_format: DataFormat,
    model: str = "gemini-2.5-flash",
    temperature: float = 1.0,
    thinking_enabled: bool = False
) -> ExperimentResult:
    """Run a single experiment with one question and one data format."""
    client = genai.Client()
    
    data_prompt = _prepare_data_prompt(data_format)
    
    system_prompt = """You are a data analyst assistant. Answer the question based solely on the provided data.
Be precise and concise. If you need to perform calculations, show your reasoning briefly.
Give a direct answer first, then explain if needed."""

    full_prompt = f"""{data_prompt}

---

Question: {question}

Please provide your answer:"""

    logger.info(f"Running experiment: {data_format.value} | Question: {question[:50]}...")
    
    start_time = time.perf_counter()
    
    # Build config
    config = {
        "system_instruction": system_prompt,
        "temperature": temperature,
    }
    
    # Enable thinking for 2.5 models if requested
    if thinking_enabled and "2.5" in model:
        config["thinking_config"] = {"thinking_budget": 4096}
    
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=config
    )
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    # Extract text from response - try multiple methods
    answer = ""
    try:
        # First try the .text property
        if response.text:
            answer = response.text
        # Fall back to extracting from candidates/parts
        elif response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            answer += part.text
    except Exception as e:
        logger.warning(f"Error extracting text from response for {data_format.value}: {e}")
        # Log the response structure for debugging
        if response.candidates:
            for i, candidate in enumerate(response.candidates):
                logger.warning(f"Candidate {i}: finish_reason={getattr(candidate, 'finish_reason', 'N/A')}")
                if candidate.content and candidate.content.parts:
                    for j, part in enumerate(candidate.content.parts):
                        logger.warning(f"  Part {j}: type={type(part).__name__}, has_text={hasattr(part, 'text')}")
        answer = ""
    
    if not answer:
        logger.warning(f"Empty response for {data_format.value}")
        # More debug info
        if response.candidates:
            for i, candidate in enumerate(response.candidates):
                logger.warning(f"Debug - Candidate {i}: finish_reason={getattr(candidate, 'finish_reason', 'N/A')}")
                if candidate.content:
                    logger.warning(f"  Content role={getattr(candidate.content, 'role', 'N/A')}")
                    if candidate.content.parts:
                        for j, part in enumerate(candidate.content.parts):
                            logger.warning(f"    Part {j}: {type(part).__name__}")
                            if hasattr(part, 'thought') and part.thought:
                                logger.warning(f"    Has thought content")
                            if hasattr(part, 'text'):
                                logger.warning(f"    Text (first 100 chars): {str(part.text)[:100] if part.text else 'EMPTY'}")
    
    # Extract token usage from response
    input_tokens = 0
    output_tokens = 0
    if response.usage_metadata:
        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0
    
    return ExperimentResult(
        question=question,
        data_format=data_format,
        answer=answer,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        system_prompt=system_prompt,
        user_prompt=full_prompt,
        raw_response={}
    )


def run_single_experiment_streaming(
    question: str,
    data_format: DataFormat,
    model: str = "gemini-2.5-flash",
    temperature: float = 1.0,
    thinking_enabled: bool = False
) -> Generator[dict, None, None]:
    """Run a single experiment with streaming response."""
    client = genai.Client()
    
    data_prompt = _prepare_data_prompt(data_format)
    
    system_prompt = """You are a data analyst assistant. Answer the question based solely on the provided data.
Be precise and concise. If you need to perform calculations, show your reasoning briefly.
Give a direct answer first, then explain if needed."""

    full_prompt = f"""{data_prompt}

---

Question: {question}

Please provide your answer:"""

    logger.info(f"Running streaming experiment: {data_format.value} | Model: {model} | Temp: {temperature} | Thinking: {thinking_enabled}")
    
    # Send the prompt info first
    yield {
        "type": "prompt",
        "data_format": data_format.value,
        "system_prompt": system_prompt,
        "user_prompt": full_prompt,
        "prompt_length": len(full_prompt),
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    full_answer = ""
    input_tokens = 0
    output_tokens = 0
    thinking_content = ""
    
    # Build config
    config = {
        "system_instruction": system_prompt,
        "temperature": temperature,
    }
    
    # Add thinking config if enabled (for 2.5 models)
    if thinking_enabled and "2.5" in model:
        config["thinking_config"] = {"thinking_budget": 4096}
    
    try:
        # Stream the response
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=full_prompt,
            config=config
        ):
            # Try to extract text from chunk - handle various response formats
            chunk_text = None
            chunk_thinking = None
            
            try:
                # Check for candidates with parts (handles thinking and regular content)
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                # Check for thinking content
                                if hasattr(part, 'thought') and part.thought:
                                    chunk_thinking = part.text if hasattr(part, 'text') else None
                                # Check for regular text
                                elif hasattr(part, 'text') and part.text:
                                    chunk_text = part.text
                
                # Fallback to direct text access
                if not chunk_text:
                    try:
                        chunk_text = chunk.text
                    except (AttributeError, ValueError, IndexError):
                        pass
                        
            except Exception as e:
                logger.debug(f"Error extracting chunk content: {e}")
            
            # Handle thinking content
            if chunk_thinking:
                thinking_content += chunk_thinking
                yield {
                    "type": "thinking",
                    "data_format": data_format.value,
                    "text": chunk_thinking,
                }
            
            if chunk_text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_answer += chunk_text
                yield {
                    "type": "chunk",
                    "data_format": data_format.value,
                    "text": chunk_text,
                }
            
            # Get token counts from chunk (usually in final chunk)
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                input_tokens = getattr(chunk.usage_metadata, 'prompt_token_count', 0) or 0
                output_tokens = getattr(chunk.usage_metadata, 'candidates_token_count', 0) or 0
                
    except Exception as e:
        logger.exception(f"Error during streaming for {data_format.value}")
        yield {
            "type": "error",
            "data_format": data_format.value,
            "error": str(e),
        }
        return
    
    # Log if we got no answer
    if not full_answer:
        logger.warning(f"Empty response for {data_format.value} - no text in any chunks")
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else latency_ms
    
    # Send final result
    yield {
        "type": "complete",
        "data_format": data_format.value,
        "answer": full_answer,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
    }


def run_comparison(
    question: str,
    expected_answer: str | None = None,
    model: str = "gemini-2.5-flash"
) -> ComparisonResult:
    """Run all three experiment formats for a single question."""
    results = {}
    
    for data_format in DataFormat:
        result = run_single_experiment(question, data_format, model)
        results[data_format] = result
    
    return ComparisonResult(
        question=question,
        expected_answer=expected_answer,
        results=results
    )


def run_full_experiment(
    model: str = "gemini-2.5-flash",
    questions: list[dict] | None = None
) -> list[ComparisonResult]:
    """Run the complete experiment with all questions."""
    if questions is None:
        questions = RESEARCH_QUESTIONS
    
    all_results = []
    
    for i, q in enumerate(questions, 1):
        logger.info(f"Processing question {i}/{len(questions)}: {q['question'][:50]}...")
        
        comparison = run_comparison(
            question=q["question"],
            expected_answer=q.get("expected"),
            model=model
        )
        all_results.append(comparison)
    
    return all_results


def format_results_markdown(results: list[ComparisonResult]) -> str:
    """Format experiment results as a markdown report."""
    lines = ["# Experiment Results\n"]
    
    for i, comparison in enumerate(results, 1):
        lines.append(f"\n## Question {i}: {comparison.question}\n")
        
        if comparison.expected_answer:
            lines.append(f"**Expected:** {comparison.expected_answer}\n")
        
        lines.append("\n| Format | Answer | Latency | Tokens (in/out) |")
        lines.append("|--------|--------|---------|-----------------|")
        
        for fmt in DataFormat:
            result = comparison.results[fmt]
            # Truncate answer for table display
            answer_short = result.answer[:100].replace("\n", " ")
            if len(result.answer) > 100:
                answer_short += "..."
            lines.append(
                f"| {fmt.value} | {answer_short} | {result.latency_ms:.0f}ms | "
                f"{result.input_tokens}/{result.output_tokens} |"
            )
        
        lines.append("")
    
    return "\n".join(lines)
