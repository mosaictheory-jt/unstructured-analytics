"""FastAPI web application for the research experiment."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .csv_to_english import convert_all_to_english
from .data_loader import get_all_csv_as_string, load_all_tables, load_metadata
from .experiment import (
    RESEARCH_QUESTIONS,
    ComparisonResult,
    DataFormat,
    run_comparison,
    run_single_experiment,
    run_single_experiment_streaming,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Structured Unstructured Reasoning research app...")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Structured Unstructured Reasoning",
    description="Research project comparing LLM performance with different data formats",
    version="0.1.0",
    lifespan=lifespan
)


class QuestionRequest(BaseModel):
    """Request to run an experiment with a question."""
    question: str
    model: str = "gemini-2.5-flash"
    temperature: float = 1.0
    thinking_enabled: bool = False


class SingleExperimentRequest(BaseModel):
    """Request for a single experiment run."""
    question: str
    data_format: str
    model: str = "gemini-2.5-flash"
    temperature: float = 1.0
    thinking_enabled: bool = False


# Available Google AI models
AVAILABLE_MODELS = [
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "description": "Fast and efficient"},
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "description": "Most capable"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "description": "Previous gen fast"},
    {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "description": "Stable fast model"},
    {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "description": "Stable pro model"},
]


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main application page."""
    return get_html_page()


@app.get("/api/data/preview")
async def get_data_preview():
    """Get a preview of the data in all three formats."""
    tables = load_all_tables()
    
    return {
        "raw_csv": get_all_csv_as_string(),
        "csv_with_metadata": {
            "metadata": load_metadata(),
            "csv_data": get_all_csv_as_string()
        },
        "english": convert_all_to_english(),
        "table_counts": {name: len(df) for name, df in tables.items()}
    }


@app.get("/api/data/schema")
async def get_data_schema():
    """Get the schema and metadata for all tables."""
    tables = load_all_tables()
    metadata = load_metadata()
    
    # Get the tables section from metadata
    tables_metadata = metadata.get("tables", {})
    
    schema = {}
    for table_name, df in tables.items():
        # Get field info from metadata if available
        table_meta = tables_metadata.get(table_name, {})
        meta_fields = table_meta.get("fields", {})
        
        columns = []
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
            }
            # Add metadata info if available
            if col in meta_fields:
                col_info["description"] = meta_fields[col].get("description", "")
                col_info["type"] = meta_fields[col].get("type", str(df[col].dtype))
            columns.append(col_info)
        
        schema[table_name] = {
            "columns": columns,
            "row_count": len(df),
            "description": table_meta.get("description", ""),
            "primary_key": table_meta.get("primary_key", ""),
            "foreign_keys": table_meta.get("foreign_keys", {})
        }
    
    return {"schema": schema}


@app.get("/api/data/tables/{table_name}")
async def get_table_data(table_name: str):
    """Get the data for a specific table."""
    tables = load_all_tables()
    
    if table_name not in tables:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
    
    df = tables[table_name]
    return {
        "table_name": table_name,
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
        "row_count": len(df)
    }


class SQLQueryRequest(BaseModel):
    """Request to execute a SQL query."""
    query: str


@app.post("/api/data/query")
async def execute_sql_query(request: SQLQueryRequest):
    """Execute a SQL query against the sample data."""
    import duckdb
    
    tables = load_all_tables()
    
    try:
        # Create a DuckDB connection and register all tables
        conn = duckdb.connect(":memory:")
        for table_name, df in tables.items():
            conn.register(table_name, df)
        
        # Execute the query
        result = conn.execute(request.query).fetchdf()
        
        return {
            "success": True,
            "columns": list(result.columns),
            "data": result.to_dict(orient="records"),
            "row_count": len(result)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if 'conn' in locals():
            conn.close()


@app.get("/api/questions")
async def get_research_questions():
    """Get the predefined research questions."""
    return {"questions": RESEARCH_QUESTIONS}


@app.get("/api/models")
async def get_available_models():
    """Get the list of available models."""
    return {"models": AVAILABLE_MODELS}


@app.post("/api/experiment/single")
async def run_single(request: SingleExperimentRequest):
    """Run a single experiment with one format."""
    try:
        data_format = DataFormat(request.data_format)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid data format: {request.data_format}")
    
    try:
        result = run_single_experiment(
            question=request.question,
            data_format=data_format,
            model=request.model
        )
        return {
            "question": result.question,
            "data_format": result.data_format.value,
            "answer": result.answer,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "model": result.model
        }
    except Exception as e:
        logger.exception("Error running experiment")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiment/compare")
async def run_comparison_endpoint(request: QuestionRequest):
    """Run a comparison experiment with all three formats."""
    try:
        comparison = run_comparison(
            question=request.question,
            model=request.model
        )
        return comparison.to_dict()
    except Exception as e:
        logger.exception("Error running comparison")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiment/parallel")
async def run_parallel_experiment(request: QuestionRequest):
    """Run comparison experiment with all three formats in parallel."""
    import concurrent.futures
    
    formats = [DataFormat.RAW_CSV, DataFormat.CSV_WITH_METADATA, DataFormat.ENGLISH_SENTENCES]
    results = {}
    
    def run_format(data_format: DataFormat):
        """Run a single format experiment."""
        try:
            result = run_single_experiment(
                question=request.question,
                data_format=data_format,
                model=request.model,
                temperature=request.temperature,
                thinking_enabled=request.thinking_enabled
            )
            return data_format.value, {
                "data_format": result.data_format.value,
                "answer": result.answer,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "model": result.model,
                "system_prompt": result.system_prompt,
                "user_prompt": result.user_prompt,
            }
        except Exception as e:
            logger.exception(f"Error running {data_format.value}")
            return data_format.value, {"error": str(e)}
    
    # Run all formats in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_format, fmt): fmt for fmt in formats}
        for future in concurrent.futures.as_completed(futures):
            fmt_value, result = future.result()
            results[fmt_value] = result
    
    return {
        "question": request.question,
        "model": request.model,
        "results": results
    }


def get_html_page() -> str:
    """Generate the HTML page for the application."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Structured Unstructured Reasoning | LLM Data Format Research</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-deep: #0a0a0f;
            --bg-card: #12121a;
            --bg-elevated: #1a1a25;
            --border-subtle: #2a2a3a;
            --border-accent: #3d3d52;
            --text-primary: #e8e8ed;
            --text-secondary: #8888a0;
            --text-muted: #5a5a70;
            --accent-cyan: #00d4ff;
            --accent-magenta: #ff006e;
            --accent-gold: #ffd60a;
            --accent-emerald: #00ff88;
            --gradient-cyan: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            --gradient-magenta: linear-gradient(135deg, #ff006e 0%, #cc0055 100%);
            --gradient-gold: linear-gradient(135deg, #ffd60a 0%, #cc9900 100%);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-deep);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .app-container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        /* Header */
        .header {
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem 0;
            position: relative;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 600px;
            height: 300px;
            background: radial-gradient(ellipse, rgba(0, 212, 255, 0.08) 0%, transparent 70%);
            pointer-events: none;
        }
        
        .logo {
            font-size: 3.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-magenta) 50%, var(--accent-gold) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            position: relative;
        }
        
        .tagline {
            font-size: 1.1rem;
            color: var(--text-secondary);
            font-weight: 400;
        }
        
        /* Cards Grid */
        .experiment-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: var(--border-accent);
            transform: translateY(-2px);
        }
        
        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-subtle);
        }
        
        .card-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
        
        .card-icon.csv { background: var(--gradient-cyan); }
        .card-icon.meta { background: var(--gradient-magenta); }
        .card-icon.english { background: var(--gradient-gold); }
        
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .card-subtitle {
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        
        /* Question Input Section */
        .question-section {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .input-group {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .question-input {
            flex: 1;
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            font-family: 'Outfit', sans-serif;
            font-size: 1rem;
            color: var(--text-primary);
            transition: all 0.2s ease;
        }
        
        .question-input:focus {
            outline: none;
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
        }
        
        .question-input::placeholder {
            color: var(--text-muted);
        }
        
        .run-btn {
            background: var(--gradient-cyan);
            border: none;
            border-radius: 12px;
            padding: 1rem 2rem;
            font-family: 'Outfit', sans-serif;
            font-size: 1rem;
            font-weight: 600;
            color: var(--bg-deep);
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .run-btn:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
        }
        
        .run-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        /* Preset Questions */
        .preset-questions {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 1rem;
        }
        
        .difficulty-group {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: center;
        }
        
        .difficulty-label {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            min-width: 60px;
            text-align: center;
        }
        
        .difficulty-label.easy {
            background: rgba(0, 255, 136, 0.15);
            color: var(--accent-emerald);
        }
        
        .difficulty-label.medium {
            background: rgba(255, 214, 10, 0.15);
            color: var(--accent-gold);
        }
        
        .difficulty-label.hard {
            background: rgba(255, 0, 110, 0.15);
            color: var(--accent-magenta);
        }
        
        .preset-btn {
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-family: 'Outfit', sans-serif;
            font-size: 0.85rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .preset-btn:hover {
            border-color: var(--accent-cyan);
            color: var(--text-primary);
        }
        
        .preset-btn.easy:hover { border-color: var(--accent-emerald); }
        .preset-btn.medium:hover { border-color: var(--accent-gold); }
        .preset-btn.hard:hover { border-color: var(--accent-magenta); }
        
        /* Results Section */
        .results-section {
            display: none;
        }
        
        .results-section.visible {
            display: block;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        
        .result-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            margin-bottom: 1rem;
            overflow: hidden;
        }
        
        .result-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--border-subtle);
        }
        
        .result-format {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .format-badge {
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .format-badge.csv {
            background: rgba(100, 149, 237, 0.15);
            color: #6495ed;
        }
        
        .format-badge.meta {
            background: rgba(147, 112, 219, 0.15);
            color: #9370db;
        }
        
        .format-badge.english {
            background: rgba(64, 224, 208, 0.15);
            color: #40e0d0;
        }
        
        .result-stats {
            display: flex;
            gap: 1.5rem;
        }
        
        .stat {
            text-align: right;
        }
        
        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            font-weight: 600;
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .result-answer {
            padding: 1.5rem;
            background: var(--bg-elevated);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            line-height: 1.7;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
        
        /* Loading State */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            color: var(--text-secondary);
        }
        
        .spinner {
            width: 24px;
            height: 24px;
            border: 3px solid var(--border-subtle);
            border-top-color: var(--accent-cyan);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 1rem;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .cursor-blink {
            animation: blink 1s step-end infinite;
            color: var(--accent-cyan);
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }
        
        .streaming-indicator {
            display: inline-flex;
            align-items: center;
            margin-left: 0.5rem;
        }
        
        /* Settings Panel */
        .settings-panel {
            margin-top: 1.5rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            overflow: hidden;
        }
        
        .settings-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            cursor: pointer;
            color: var(--text-secondary);
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }
        
        .settings-header:hover {
            color: var(--text-primary);
            background: rgba(255,255,255,0.02);
        }
        
        .settings-content {
            display: none;
            padding: 1rem;
            border-top: 1px solid var(--border-subtle);
        }
        
        .settings-content.visible {
            display: block;
        }
        
        .setting-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .setting-row:last-child {
            margin-bottom: 0;
        }
        
        .setting-row label {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .setting-select {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            font-size: 0.9rem;
            min-width: 200px;
            cursor: pointer;
        }
        
        .setting-select:focus {
            outline: none;
            border-color: var(--accent-cyan);
        }
        
        .setting-slider {
            width: 200px;
            height: 6px;
            -webkit-appearance: none;
            background: var(--border-subtle);
            border-radius: 3px;
            cursor: pointer;
        }
        
        .setting-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--accent-cyan);
            border-radius: 50%;
            cursor: pointer;
        }
        
        .setting-checkbox {
            width: 18px;
            height: 18px;
            accent-color: var(--accent-cyan);
            cursor: pointer;
            margin-right: 0.5rem;
        }
        
        #temp-value {
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Thinking indicator */
        .thinking-badge {
            background: rgba(255, 214, 10, 0.15);
            color: var(--accent-gold);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            margin-left: 0.5rem;
        }
        
        /* Prompt viewer */
        .prompt-toggle {
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 0.35rem 0.75rem;
            font-family: 'Outfit', sans-serif;
            font-size: 0.8rem;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
            margin-left: auto;
        }
        
        .prompt-toggle:hover {
            border-color: var(--accent-cyan);
            color: var(--text-primary);
        }
        
        .prompt-section {
            display: none;
            border-top: 1px solid var(--border-subtle);
            padding: 1rem;
            background: var(--bg-deep);
        }
        
        .prompt-section.visible {
            display: block;
        }
        
        .prompt-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }
        
        .prompt-content {
            background: var(--bg-elevated);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            line-height: 1.5;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }
        
        .prompt-content:last-child {
            margin-bottom: 0;
        }
        
        .prompt-stats {
            display: flex;
            gap: 1rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }
        
        .prompt-stats span {
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Data Explorer */
        .data-explorer {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            margin-bottom: 2rem;
            overflow: hidden;
        }
        
        .explorer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-subtle);
        }
        
        .explorer-toggle {
            background: none;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-family: 'Outfit', sans-serif;
            font-size: 0.9rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .explorer-toggle:hover {
            border-color: var(--accent-cyan);
            color: var(--text-primary);
        }
        
        .explorer-content {
            display: none;
        }
        
        .explorer-content.visible {
            display: block;
        }
        
        .explorer-layout {
            display: grid;
            grid-template-columns: 280px 1fr;
            min-height: 500px;
        }
        
        /* Schema Panel (Left) */
        .schema-panel {
            background: var(--bg-deep);
            border-right: 1px solid var(--border-subtle);
            padding: 1rem;
            overflow-y: auto;
            max-height: 600px;
        }
        
        .schema-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-subtle);
        }
        
        .table-item {
            margin-bottom: 0.5rem;
        }
        
        .table-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            background: var(--bg-elevated);
            border: 1px solid transparent;
        }
        
        .table-header:hover {
            border-color: var(--border-accent);
        }
        
        .table-header.active {
            border-color: var(--accent-cyan);
            background: rgba(0, 212, 255, 0.08);
        }
        
        .table-icon {
            font-size: 0.9rem;
        }
        
        .table-name {
            flex: 1;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-primary);
        }
        
        .table-count {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .table-columns {
            display: none;
            padding: 0.5rem 0 0.5rem 1.5rem;
        }
        
        .table-columns.visible {
            display: block;
        }
        
        .column-item {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            padding: 0.4rem 0.5rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            border-radius: 4px;
            border-left: 2px solid transparent;
            margin-bottom: 0.25rem;
        }
        
        .column-item:hover {
            background: rgba(255,255,255,0.03);
            border-left-color: var(--accent-cyan);
        }
        
        .column-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .column-name {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-cyan);
            font-size: 0.8rem;
        }
        
        .column-type {
            font-size: 0.65rem;
            color: var(--text-muted);
            margin-left: auto;
            background: var(--bg-elevated);
            padding: 0.1rem 0.4rem;
            border-radius: 3px;
        }
        
        .column-desc {
            font-size: 0.7rem;
            color: var(--text-muted);
            font-style: italic;
            line-height: 1.3;
            padding-left: 0.25rem;
        }
        
        /* Data Panel (Right) */
        .data-panel {
            display: flex;
            flex-direction: column;
        }
        
        .data-tabs {
            display: flex;
            gap: 0;
            background: var(--bg-elevated);
            border-bottom: 1px solid var(--border-subtle);
        }
        
        .data-tab {
            background: none;
            border: none;
            padding: 0.75rem 1.25rem;
            font-family: 'Outfit', sans-serif;
            font-size: 0.85rem;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
        }
        
        .data-tab:hover {
            color: var(--text-secondary);
        }
        
        .data-tab.active {
            color: var(--accent-cyan);
            border-bottom-color: var(--accent-cyan);
        }
        
        .table-container {
            flex: 1;
            overflow: auto;
            padding: 1rem;
            display: none;
        }
        
        .table-container.visible {
            display: block;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }
        
        .data-table th {
            background: var(--bg-elevated);
            padding: 0.75rem 1rem;
            text-align: left;
            font-weight: 500;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-subtle);
            position: sticky;
            top: 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        
        .data-table td {
            padding: 0.6rem 1rem;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }
        
        .data-table tr:hover td {
            background: rgba(0, 212, 255, 0.03);
        }
        
        /* SQL Panel */
        .sql-container {
            flex: 1;
            display: none;
            flex-direction: column;
            padding: 1rem;
        }
        
        .sql-container.visible {
            display: flex;
        }
        
        .sql-editor-wrapper {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .sql-editor {
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-primary);
            resize: vertical;
            min-height: 100px;
            line-height: 1.5;
        }
        
        .sql-editor:focus {
            outline: none;
            border-color: var(--accent-cyan);
        }
        
        .sql-editor::placeholder {
            color: var(--text-muted);
        }
        
        .sql-actions {
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }
        
        .sql-run-btn {
            background: var(--gradient-cyan);
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.25rem;
            font-family: 'Outfit', sans-serif;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--bg-deep);
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        
        .sql-run-btn:hover {
            transform: scale(1.02);
            box-shadow: 0 2px 12px rgba(0, 212, 255, 0.3);
        }
        
        .sql-examples {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        
        .sql-example-btn {
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 0.35rem 0.75rem;
            font-family: 'Outfit', sans-serif;
            font-size: 0.75rem;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .sql-example-btn:hover {
            border-color: var(--accent-cyan);
            color: var(--text-secondary);
        }
        
        .sql-results {
            flex: 1;
            overflow: auto;
            background: var(--bg-elevated);
            border-radius: 8px;
            border: 1px solid var(--border-subtle);
        }
        
        .sql-results-header {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .sql-results-title {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        
        .sql-results-count {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--accent-emerald);
        }
        
        .sql-results-body {
            overflow: auto;
            max-height: 300px;
        }
        
        .sql-error {
            padding: 1rem;
            color: var(--accent-magenta);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }
        
        .sql-empty {
            padding: 2rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        
        /* ERD Styles */
        .erd-container {
            flex: 1;
            display: none;
            padding: 1.5rem;
            overflow: auto;
            background: var(--bg-deep);
        }
        
        .erd-container.visible {
            display: block;
        }
        
        .erd-canvas {
            position: relative;
            min-height: 700px;
            min-width: 1000px;
        }
        
        .erd-svg {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }
        
        .erd-table {
            position: absolute;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            min-width: 180px;
            overflow: hidden;
            z-index: 2;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .erd-table:hover {
            border-color: var(--accent-cyan);
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.15);
        }
        
        .erd-table-header {
            background: var(--bg-elevated);
            padding: 0.6rem 0.8rem;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .erd-table-icon {
            font-size: 0.9rem;
        }
        
        .erd-table-name {
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--text-primary);
        }
        
        .erd-table-count {
            margin-left: auto;
            font-size: 0.65rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .erd-columns {
            padding: 0.4rem 0;
        }
        
        .erd-column {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.8rem;
            font-size: 0.75rem;
        }
        
        .erd-column:hover {
            background: rgba(255,255,255,0.02);
        }
        
        .erd-key-icon {
            font-size: 0.65rem;
            width: 14px;
        }
        
        .erd-key-icon.pk {
            color: var(--accent-gold);
        }
        
        .erd-key-icon.fk {
            color: var(--accent-cyan);
        }
        
        .erd-col-name {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
            font-size: 0.7rem;
        }
        
        .erd-col-name.pk {
            color: var(--accent-gold);
        }
        
        .erd-col-name.fk {
            color: var(--accent-cyan);
        }
        
        .erd-col-type {
            margin-left: auto;
            font-size: 0.6rem;
            color: var(--text-muted);
            background: var(--bg-elevated);
            padding: 0.1rem 0.35rem;
            border-radius: 3px;
        }
        
        .erd-legend {
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-size: 0.7rem;
            z-index: 3;
        }
        
        .erd-legend-title {
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .erd-legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
            color: var(--text-muted);
        }
        
        .erd-legend-item svg {
            width: 40px;
            height: 16px;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        
        /* Responsive */
        @media (max-width: 1200px) {
            .experiment-grid {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .app-container {
                padding: 1rem;
            }
            
            .logo {
                font-size: 2.5rem;
            }
            
            .input-group {
                flex-direction: column;
            }
            
            .result-card-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 1rem;
            }
            
            .result-stats {
                width: 100%;
                justify-content: space-between;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <header class="header">
            <h1 class="logo">Structured Unstructured Reasoning</h1>
            <p class="tagline">Comparing LLM performance: CSV vs CSV + Metadata vs Natural Language</p>
        </header>
        
        <section class="data-explorer">
            <div class="explorer-header">
                <h2 class="section-title" style="margin:0;">📊 Data Explorer</h2>
                <button class="explorer-toggle" onclick="toggleExplorer()">Show Data Explorer</button>
            </div>
            <div id="explorer-content" class="explorer-content">
                <div class="explorer-layout">
                    <div class="schema-panel" id="schema-panel">
                        <div class="schema-title">Schema & Metadata</div>
                        <div id="schema-tree">Loading...</div>
                    </div>
                    <div class="data-panel">
                        <div class="data-tabs">
                            <button class="data-tab active" onclick="showDataTab('tables')" id="tab-tables">📋 Tables</button>
                            <button class="data-tab" onclick="showDataTab('erd')" id="tab-erd">🔗 ERD</button>
                            <button class="data-tab" onclick="showDataTab('sql')" id="tab-sql">🔍 SQL Query</button>
                        </div>
                        <div class="table-container visible" id="table-view">
                            <div id="table-content" style="text-align:center;color:var(--text-muted);padding:2rem;">
                                Select a table from the schema panel to view its data
                            </div>
                        </div>
                        <div class="erd-container" id="erd-view">
                            <div id="erd-content"></div>
                        </div>
                        <div class="sql-container" id="sql-view">
                            <div class="sql-editor-wrapper">
                                <textarea class="sql-editor" id="sql-input" placeholder="SELECT * FROM products LIMIT 10;"></textarea>
                                <div class="sql-actions">
                                    <button class="sql-run-btn" onclick="runSQL()">▶ Run Query</button>
                                    <span style="color:var(--text-muted);font-size:0.8rem;">or press Ctrl+Enter</span>
                                    <div class="sql-examples" style="margin-left:auto;">
                                        <button class="sql-example-btn" onclick="setSQLExample('SELECT * FROM products ORDER BY unit_price DESC LIMIT 5')">Top Products</button>
                                        <button class="sql-example-btn" onclick="setSQLExample('SELECT c.first_name, c.last_name, COUNT(o.order_id) as order_count FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id ORDER BY order_count DESC')">Orders by Customer</button>
                                        <button class="sql-example-btn" onclick="setSQLExample('SELECT p.category, SUM(oi.total_price) as revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.category ORDER BY revenue DESC')">Revenue by Category</button>
                                    </div>
                                </div>
                            </div>
                            <div class="sql-results" id="sql-results">
                                <div class="sql-empty">Run a query to see results</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="question-section">
            <h2 class="section-title">🔬 Run Experiment</h2>
            <div class="input-group">
                <input type="text" id="question-input" class="question-input" 
                       placeholder="Ask a question about the e-commerce data...">
                <button class="run-btn" onclick="runExperiment()" id="run-btn">
                    <span>▶</span> Run Comparison
                </button>
            </div>
            <div class="preset-questions" id="preset-questions">
                <!-- Filled by JS -->
            </div>
            
            <div class="settings-panel">
                <div class="settings-header" onclick="toggleSettings()">
                    <span>⚙️ Model Settings</span>
                    <span id="settings-toggle">▼</span>
                </div>
                <div class="settings-content" id="settings-content">
                    <div class="setting-row">
                        <label for="model-select">Model</label>
                        <select id="model-select" class="setting-select">
                            <!-- Filled by JS -->
                        </select>
                    </div>
                    <div class="setting-row">
                        <label for="temperature-slider">Temperature: <span id="temp-value">1.0</span></label>
                        <input type="range" id="temperature-slider" class="setting-slider" 
                               min="0" max="2" step="0.1" value="1.0"
                               oninput="document.getElementById('temp-value').textContent = this.value">
                    </div>
                    <div class="setting-row">
                        <label for="thinking-toggle">
                            <input type="checkbox" id="thinking-toggle" class="setting-checkbox">
                            Enable Thinking (2.5 models only)
                        </label>
                    </div>
                </div>
            </div>
        </section>
        
        <section class="results-section" id="results-section">
            <div class="results-header">
                <h2 class="section-title">📈 Results</h2>
                <span id="current-question" style="color: var(--text-secondary);"></span>
            </div>
            <div id="results-container">
                <!-- Filled by JS -->
            </div>
        </section>
        
        <footer class="footer">
            Research project exploring how data representation affects LLM comprehension and accuracy.
        </footer>
    </div>
    
    <script>
        let schemaData = null;
        let availableModels = [];
        let activeTable = null;
        
        async function init() {
            // Load preset questions grouped by difficulty
            const resp = await fetch('/api/questions');
            const data = await resp.json();
            
            const container = document.getElementById('preset-questions');
            const byDifficulty = {easy: [], medium: [], hard: []};
            data.questions.forEach(q => {
                const diff = q.difficulty || 'medium';
                if (byDifficulty[diff]) byDifficulty[diff].push(q);
            });
            
            container.innerHTML = Object.entries(byDifficulty).map(([diff, questions]) => `
                <div class="difficulty-group">
                    <span class="difficulty-label ${diff}">${diff}</span>
                    ${questions.slice(0, 3).map(q => 
                        `<button class="preset-btn ${diff}" onclick="setQuestion('${q.question.replace(/'/g, "\\\\'")}')">${q.question.substring(0, 45)}${q.question.length > 45 ? '...' : ''}</button>`
                    ).join('')}
                </div>
            `).join('');
            
            // Load schema
            const schemaResp = await fetch('/api/data/schema');
            const schemaResult = await schemaResp.json();
            schemaData = schemaResult.schema;
            renderSchemaTree();
            
            // Load available models
            const modelsResp = await fetch('/api/models');
            const modelsData = await modelsResp.json();
            availableModels = modelsData.models;
            
            const modelSelect = document.getElementById('model-select');
            modelSelect.innerHTML = availableModels.map(m => 
                `<option value="${m.id}" ${m.id === 'gemini-2.5-flash' ? 'selected' : ''}>${m.name} - ${m.description}</option>`
            ).join('');
            
            // SQL editor keyboard shortcut
            document.getElementById('sql-input').addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') {
                    runSQL();
                }
            });
        }
        
        function renderSchemaTree() {
            if (!schemaData) return;
            
            const container = document.getElementById('schema-tree');
            container.innerHTML = Object.entries(schemaData).map(([tableName, info]) => `
                <div class="table-item">
                    <div class="table-header" onclick="toggleTableSchema('${tableName}')" id="table-header-${tableName}">
                        <span class="table-icon">📁</span>
                        <span class="table-name">${tableName}</span>
                        <span class="table-count">${info.row_count} rows</span>
                    </div>
                    <div class="table-columns" id="columns-${tableName}">
                        ${info.description ? `<div style="padding:0.5rem 0.5rem 0.75rem;font-size:0.75rem;color:var(--text-secondary);border-bottom:1px solid var(--border-subtle);margin-bottom:0.5rem;">${info.description}</div>` : ''}
                        ${info.columns.map(col => `
                            <div class="column-item">
                                <div class="column-header">
                                    <span class="column-name">${col.name}</span>
                                    <span class="column-type">${col.type || col.dtype}</span>
                                </div>
                                ${col.description ? `<div class="column-desc">${col.description}</div>` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            `).join('');
        }
        
        function toggleTableSchema(tableName) {
            const columns = document.getElementById(`columns-${tableName}`);
            const header = document.getElementById(`table-header-${tableName}`);
            
            // Toggle column visibility
            columns.classList.toggle('visible');
            
            // Select table and show data
            document.querySelectorAll('.table-header').forEach(h => h.classList.remove('active'));
            header.classList.add('active');
            activeTable = tableName;
            loadTableData(tableName);
            
            // Switch to tables tab
            showDataTab('tables');
        }
        
        async function loadTableData(tableName) {
            const container = document.getElementById('table-content');
            container.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--text-muted);">Loading...</div>';
            
            try {
                const resp = await fetch(`/api/data/tables/${tableName}`);
                const data = await resp.json();
                
                container.innerHTML = `
                    <table class="data-table">
                        <thead>
                            <tr>${data.columns.map(col => `<th>${col}</th>`).join('')}</tr>
                        </thead>
                        <tbody>
                            ${data.data.map(row => `
                                <tr>${data.columns.map(col => `<td>${row[col] ?? ''}</td>`).join('')}</tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            } catch (err) {
                container.innerHTML = `<div style="text-align:center;padding:2rem;color:var(--accent-magenta);">Error loading table: ${err.message}</div>`;
            }
        }
        
        function showDataTab(tab) {
            document.querySelectorAll('.data-tab').forEach(t => t.classList.remove('active'));
            document.getElementById(`tab-${tab}`).classList.add('active');
            
            document.getElementById('table-view').classList.toggle('visible', tab === 'tables');
            document.getElementById('erd-view').classList.toggle('visible', tab === 'erd');
            document.getElementById('sql-view').classList.toggle('visible', tab === 'sql');
            
            if (tab === 'erd' && !document.getElementById('erd-content').innerHTML) {
                renderERD();
            }
        }
        
        function renderERD() {
            if (!schemaData) return;
            
            const relationships = [
                {from: 'orders', fromCol: 'customer_id', to: 'customers', toCol: 'customer_id', type: 'N:1'},
                {from: 'order_items', fromCol: 'order_id', to: 'orders', toCol: 'order_id', type: 'N:1'},
                {from: 'order_items', fromCol: 'product_id', to: 'products', toCol: 'product_id', type: 'N:1'},
                {from: 'products', fromCol: 'supplier_id', to: 'suppliers', toCol: 'supplier_id', type: 'N:1'},
            ];
            
            const fkMap = {};
            relationships.forEach(r => {
                if (!fkMap[r.from]) fkMap[r.from] = [];
                fkMap[r.from].push(r.fromCol);
            });
            
            // Table positions for the ERD layout - spread out for clear relationship visibility
            const positions = {
                'customers': {x: 20, y: 150},
                'orders': {x: 280, y: 150},
                'order_items': {x: 540, y: 150},
                'products': {x: 540, y: 400},
                'suppliers': {x: 800, y: 400},
            };
            
            const tableOrder = ['customers', 'orders', 'order_items', 'products', 'suppliers'];
            
            // Build tables HTML
            const tablesHtml = tableOrder.map(tableName => {
                const info = schemaData[tableName];
                if (!info) return '';
                const pk = info.primary_key;
                const fks = fkMap[tableName] || [];
                const pos = positions[tableName];
                
                return `
                    <div class="erd-table" id="erd-table-${tableName}" style="left:${pos.x}px;top:${pos.y}px;">
                        <div class="erd-table-header">
                            <span class="erd-table-icon">📋</span>
                            <span class="erd-table-name">${tableName}</span>
                            <span class="erd-table-count">${info.row_count}</span>
                        </div>
                        <div class="erd-columns">
                            ${info.columns.map(col => {
                                const isPK = col.name === pk;
                                const isFK = fks.includes(col.name);
                                return `
                                    <div class="erd-column" data-col="${col.name}">
                                        <span class="erd-key-icon ${isPK ? 'pk' : isFK ? 'fk' : ''}">${isPK ? '🔑' : isFK ? '🔗' : ''}</span>
                                        <span class="erd-col-name ${isPK ? 'pk' : isFK ? 'fk' : ''}">${col.name}</span>
                                        <span class="erd-col-type">${col.type || col.dtype}</span>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                `;
            }).join('');
            
            // Legend HTML
            const legendHtml = `
                <div class="erd-legend">
                    <div class="erd-legend-title">Notation</div>
                    <div class="erd-legend-item">
                        <svg viewBox="0 0 40 16">
                            <line x1="0" y1="8" x2="30" y2="8" stroke="#00d4ff" stroke-width="2"/>
                            <line x1="30" y1="3" x2="38" y2="8" stroke="#00d4ff" stroke-width="2"/>
                            <line x1="30" y1="13" x2="38" y2="8" stroke="#00d4ff" stroke-width="2"/>
                        </svg>
                        One (1)
                    </div>
                    <div class="erd-legend-item">
                        <svg viewBox="0 0 40 16">
                            <line x1="10" y1="8" x2="40" y2="8" stroke="#00d4ff" stroke-width="2"/>
                            <line x1="0" y1="3" x2="10" y2="8" stroke="#00d4ff" stroke-width="2"/>
                            <line x1="0" y1="13" x2="10" y2="8" stroke="#00d4ff" stroke-width="2"/>
                            <line x1="0" y1="8" x2="10" y2="8" stroke="#00d4ff" stroke-width="2"/>
                        </svg>
                        Many (N)
                    </div>
                    <div class="erd-legend-item" style="margin-top:0.5rem;">
                        <span style="color:var(--accent-gold);">🔑</span> Primary Key
                    </div>
                    <div class="erd-legend-item">
                        <span style="color:var(--accent-cyan);">🔗</span> Foreign Key
                    </div>
                </div>
            `;
            
            const erdHtml = `
                <div class="erd-canvas">
                    <svg class="erd-svg" id="erd-svg"></svg>
                    ${tablesHtml}
                    ${legendHtml}
                </div>
            `;
            
            document.getElementById('erd-content').innerHTML = erdHtml;
            
            // Draw relationship lines after DOM is updated
            setTimeout(() => drawERDLines(relationships), 50);
        }
        
        function drawERDLines(relationships) {
            const svg = document.getElementById('erd-svg');
            if (!svg) return;
            
            const canvas = svg.parentElement;
            const canvasRect = canvas.getBoundingClientRect();
            
            let paths = '';
            
            relationships.forEach(rel => {
                const fromTable = document.getElementById(`erd-table-${rel.from}`);
                const toTable = document.getElementById(`erd-table-${rel.to}`);
                
                if (!fromTable || !toTable) return;
                
                const fromRect = fromTable.getBoundingClientRect();
                const toRect = toTable.getBoundingClientRect();
                
                // Calculate connection points
                let fromX, fromY, toX, toY;
                
                // Determine which sides to connect based on positions
                const fromCenterX = fromRect.left + fromRect.width / 2 - canvasRect.left;
                const fromCenterY = fromRect.top + fromRect.height / 2 - canvasRect.top;
                const toCenterX = toRect.left + toRect.width / 2 - canvasRect.left;
                const toCenterY = toRect.top + toRect.height / 2 - canvasRect.top;
                
                // Connect from right of "from" table to left of "to" table (or adjust based on position)
                if (fromCenterX < toCenterX) {
                    // From is to the left of To
                    fromX = fromRect.right - canvasRect.left;
                    fromY = fromRect.top + fromRect.height / 2 - canvasRect.top;
                    toX = toRect.left - canvasRect.left;
                    toY = toRect.top + toRect.height / 2 - canvasRect.top;
                } else {
                    // From is to the right of To
                    fromX = fromRect.left - canvasRect.left;
                    fromY = fromRect.top + fromRect.height / 2 - canvasRect.top;
                    toX = toRect.right - canvasRect.left;
                    toY = toRect.top + toRect.height / 2 - canvasRect.top;
                }
                
                // If tables are vertically aligned
                if (Math.abs(fromCenterX - toCenterX) < 50) {
                    if (fromCenterY < toCenterY) {
                        fromX = fromCenterX;
                        fromY = fromRect.bottom - canvasRect.top;
                        toX = toCenterX;
                        toY = toRect.top - canvasRect.top;
                    } else {
                        fromX = fromCenterX;
                        fromY = fromRect.top - canvasRect.top;
                        toX = toCenterX;
                        toY = toRect.bottom - canvasRect.top;
                    }
                }
                
                // Create path with bezier curve
                const midX = (fromX + toX) / 2;
                const pathD = `M ${fromX} ${fromY} C ${midX} ${fromY}, ${midX} ${toY}, ${toX} ${toY}`;
                
                // Draw the main line
                paths += `<path d="${pathD}" fill="none" stroke="#00d4ff" stroke-width="2" opacity="0.6"/>`;
                
                // Draw crow's foot (many) at the "from" end (N side)
                const crowAngle = Math.atan2(toY - fromY, toX - fromX);
                const crowSize = 10;
                paths += `
                    <line x1="${fromX}" y1="${fromY}" 
                          x2="${fromX + Math.cos(crowAngle + 0.4) * crowSize}" y2="${fromY + Math.sin(crowAngle + 0.4) * crowSize}" 
                          stroke="#00d4ff" stroke-width="2" opacity="0.6"/>
                    <line x1="${fromX}" y1="${fromY}" 
                          x2="${fromX + Math.cos(crowAngle - 0.4) * crowSize}" y2="${fromY + Math.sin(crowAngle - 0.4) * crowSize}" 
                          stroke="#00d4ff" stroke-width="2" opacity="0.6"/>
                    <line x1="${fromX}" y1="${fromY}" 
                          x2="${fromX + Math.cos(crowAngle) * crowSize}" y2="${fromY + Math.sin(crowAngle) * crowSize}" 
                          stroke="#00d4ff" stroke-width="2" opacity="0.6"/>
                `;
                
                // Draw single line (one) at the "to" end (1 side)
                const oneAngle = Math.atan2(fromY - toY, fromX - toX);
                const oneSize = 12;
                paths += `
                    <line x1="${toX + Math.cos(oneAngle) * 5}" y1="${toY + Math.sin(oneAngle) * 5}" 
                          x2="${toX + Math.cos(oneAngle) * 5 + Math.cos(oneAngle + Math.PI/2) * 8}" 
                          y2="${toY + Math.sin(oneAngle) * 5 + Math.sin(oneAngle + Math.PI/2) * 8}" 
                          stroke="#00d4ff" stroke-width="2" opacity="0.6"/>
                    <line x1="${toX + Math.cos(oneAngle) * 5}" y1="${toY + Math.sin(oneAngle) * 5}" 
                          x2="${toX + Math.cos(oneAngle) * 5 - Math.cos(oneAngle + Math.PI/2) * 8}" 
                          y2="${toY + Math.sin(oneAngle) * 5 - Math.sin(oneAngle + Math.PI/2) * 8}" 
                          stroke="#00d4ff" stroke-width="2" opacity="0.6"/>
                `;
            });
            
            svg.innerHTML = paths;
        }
        
        function setSQLExample(sql) {
            document.getElementById('sql-input').value = sql;
        }
        
        async function runSQL() {
            const query = document.getElementById('sql-input').value.trim();
            if (!query) return;
            
            const resultsDiv = document.getElementById('sql-results');
            resultsDiv.innerHTML = '<div class="sql-empty">Running query...</div>';
            
            try {
                const resp = await fetch('/api/data/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query})
                });
                const data = await resp.json();
                
                if (!data.success) {
                    resultsDiv.innerHTML = `<div class="sql-error">Error: ${data.error}</div>`;
                    return;
                }
                
                if (data.data.length === 0) {
                    resultsDiv.innerHTML = '<div class="sql-empty">Query returned no results</div>';
                    return;
                }
                
                resultsDiv.innerHTML = `
                    <div class="sql-results-header">
                        <span class="sql-results-title">Results</span>
                        <span class="sql-results-count">${data.row_count} row${data.row_count !== 1 ? 's' : ''}</span>
                    </div>
                    <div class="sql-results-body">
                        <table class="data-table">
                            <thead>
                                <tr>${data.columns.map(col => `<th>${col}</th>`).join('')}</tr>
                            </thead>
                            <tbody>
                                ${data.data.map(row => `
                                    <tr>${data.columns.map(col => `<td>${row[col] ?? ''}</td>`).join('')}</tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                `;
            } catch (err) {
                resultsDiv.innerHTML = `<div class="sql-error">Error: ${err.message}</div>`;
            }
        }
        
        function toggleExplorer() {
            const content = document.getElementById('explorer-content');
            const btn = document.querySelector('.explorer-toggle');
            content.classList.toggle('visible');
            btn.textContent = content.classList.contains('visible') ? 'Hide Data Explorer' : 'Show Data Explorer';
        }
        
        function toggleSettings() {
            const content = document.getElementById('settings-content');
            const toggle = document.getElementById('settings-toggle');
            content.classList.toggle('visible');
            toggle.textContent = content.classList.contains('visible') ? '▲' : '▼';
        }
        
        function getSettings() {
            return {
                model: document.getElementById('model-select').value,
                temperature: parseFloat(document.getElementById('temperature-slider').value),
                thinking_enabled: document.getElementById('thinking-toggle').checked
            };
        }
        
        function togglePrompt(format) {
            const section = document.getElementById(`prompt-section-${format}`);
            const btn = document.getElementById(`prompt-toggle-${format}`);
            if (section) {
                section.classList.toggle('visible');
                btn.textContent = section.classList.contains('visible') ? '📝 Hide Prompt' : '📝 View Prompt';
            }
        }
        
        function setQuestion(q) {
            document.getElementById('question-input').value = q;
        }
        
        const formatConfig = {
            'raw_csv': {name: 'Raw CSV', badge: 'csv'},
            'csv_with_metadata': {name: 'CSV + Metadata', badge: 'meta'},
            'english_sentences': {name: 'Natural Language', badge: 'english'}
        };
        
        async function runExperiment() {
            const question = document.getElementById('question-input').value.trim();
            if (!question) return;
            
            const btn = document.getElementById('run-btn');
            const resultsSection = document.getElementById('results-section');
            const resultsContainer = document.getElementById('results-container');
            
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner" style="width:16px;height:16px;margin:0;margin-right:8px;"></span> Running...';
            
            resultsSection.classList.add('visible');
            document.getElementById('current-question').textContent = question;
            
            // Initialize result cards with loading state
            resultsContainer.innerHTML = Object.entries(formatConfig).map(([key, fmt]) => `
                <div class="result-card" id="card-${key}">
                    <div class="result-card-header">
                        <div class="result-format">
                            <span class="format-badge ${fmt.badge}">${fmt.name}</span>
                            <span class="streaming-indicator" id="indicator-${key}">
                                <span class="spinner" style="width:14px;height:14px;margin:0;"></span>
                            </span>
                        </div>
                        <div class="result-stats" id="stats-${key}">
                            <div class="stat">
                                <div class="stat-value" id="latency-${key}">—</div>
                                <div class="stat-label">Latency</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="input-${key}">—</div>
                                <div class="stat-label">Input Tokens</div>
                            </div>
                            <div class="stat">
                                <div class="stat-value" id="output-${key}">—</div>
                                <div class="stat-label">Output Tokens</div>
                            </div>
                            <button class="prompt-toggle" id="prompt-toggle-${key}" onclick="togglePrompt('${key}')" style="display:none;">
                                📝 View Prompt
                            </button>
                        </div>
                    </div>
                    <div class="result-answer" id="answer-${key}" style="min-height: 60px;"><span style="color: var(--text-muted);">Running in parallel...</span></div>
                    <div class="prompt-section" id="prompt-section-${key}">
                        <div class="prompt-label">System Prompt</div>
                        <div class="prompt-content" id="system-prompt-${key}"></div>
                        <div class="prompt-label">User Prompt</div>
                        <div class="prompt-content" id="user-prompt-${key}"></div>
                        <div class="prompt-stats">
                            Prompt length: <span id="prompt-length-${key}">0</span> characters
                        </div>
                    </div>
                </div>
            `).join('');
            
            try {
                const settings = getSettings();
                const response = await fetch('/api/experiment/parallel', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: question,
                        model: settings.model,
                        temperature: settings.temperature,
                        thinking_enabled: settings.thinking_enabled
                    })
                });
                
                const data = await response.json();
                
                // Update each result card
                for (const [formatKey, result] of Object.entries(data.results)) {
                    const indicator = document.getElementById(`indicator-${formatKey}`);
                    const answer = document.getElementById(`answer-${formatKey}`);
                    const latency = document.getElementById(`latency-${formatKey}`);
                    const input = document.getElementById(`input-${formatKey}`);
                    const output = document.getElementById(`output-${formatKey}`);
                    const toggleBtn = document.getElementById(`prompt-toggle-${formatKey}`);
                    const systemPrompt = document.getElementById(`system-prompt-${formatKey}`);
                    const userPrompt = document.getElementById(`user-prompt-${formatKey}`);
                    const promptLength = document.getElementById(`prompt-length-${formatKey}`);
                    
                    if (indicator) indicator.style.display = 'none';
                    
                    if (result.error) {
                        if (answer) answer.innerHTML = `<span style="color: var(--accent-magenta);">Error: ${escapeHtml(result.error)}</span>`;
                    } else {
                        if (answer) answer.textContent = result.answer;
                        if (latency) latency.textContent = `${result.latency_ms.toFixed(0)}ms`;
                        if (input) input.textContent = result.input_tokens.toLocaleString();
                        if (output) output.textContent = result.output_tokens;
                        
                        // Show prompt toggle and populate prompts
                        if (toggleBtn) toggleBtn.style.display = 'block';
                        if (systemPrompt) systemPrompt.textContent = result.system_prompt;
                        if (userPrompt) userPrompt.textContent = result.user_prompt;
                        if (promptLength) promptLength.textContent = result.user_prompt.length.toLocaleString();
                    }
                }
            } catch (err) {
                console.error('Experiment error:', err);
                resultsContainer.innerHTML = `<div style="color: var(--accent-magenta); padding: 2rem;">Error: ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<span>▶</span> Run Comparison';
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Handle Enter key
        document.addEventListener('DOMContentLoaded', () => {
            init();
            document.getElementById('question-input').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') runExperiment();
            });
        });
    </script>
</body>
</html>
'''

