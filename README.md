# Structured Unstructured Reasoning

A research project comparing how Large Language Models (LLMs) perform when given the same data in different formats:

1. **Raw CSV** - Plain comma-separated values
2. **CSV with Metadata** - CSV data plus schema documentation and field descriptions  
3. **English Sentences** - The same data converted to natural language prose

## Hypothesis

LLMs may perform better when data is presented in natural English sentences rather than structured CSV format, because:

- LLMs are trained primarily on natural language text
- English sentences provide implicit context and relationships
- Structured data may require more "translation" effort by the model

## Project Structure

```
english-is-better/
├── data/                      # Sample e-commerce data (star schema)
│   ├── customers.csv          # Customer dimension
│   ├── products.csv           # Product dimension  
│   ├── suppliers.csv          # Supplier dimension
│   ├── orders.csv             # Order fact table
│   ├── order_items.csv        # Order line items
│   └── schema_metadata.json   # Field descriptions and relationships
├── src/
│   ├── data_loader.py         # CSV loading utilities
│   ├── csv_to_english.py      # Converts CSV data to English sentences
│   ├── experiment.py          # Experiment framework and questions
│   └── web_app.py             # FastAPI web application
└── pyproject.toml
```

## Quick Start

### Install Dependencies

```bash
uv sync
```

### Set API Key

Create a `.env` file with your Google API key:

```bash
GOOGLE_API_KEY="your-google-api-key"
```

### Run the Web App (Local)

```bash
uv run uvicorn src.web_app:app --reload --port 8000
```

Then open http://localhost:8000 in your browser.

## Docker Deployment

### Build and Run Locally

```bash
# Build the image
docker build -t structured-unstructured-reasoning .

# Run the container
docker run -p 8080:8080 -e GOOGLE_API_KEY="your-api-key" structured-unstructured-reasoning
```

### Deploy to Google Cloud Run

#### Option 1: Using gcloud CLI

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud run deploy structured-unstructured-reasoning \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY="your-api-key" \
  --memory 512Mi \
  --timeout 300
```

#### Option 2: Using Cloud Build

```bash
# Submit build with substitutions
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions _GOOGLE_API_KEY="your-api-key",_REGION="us-central1"
```

#### Option 3: Manual Steps

```bash
# 1. Build the image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/structured-unstructured-reasoning

# 2. Deploy to Cloud Run
gcloud run deploy structured-unstructured-reasoning \
  --image gcr.io/YOUR_PROJECT_ID/structured-unstructured-reasoning \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY="your-api-key"
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google AI API key for Gemini | Yes |
| `PORT` | Port to run the server (default: 8080) | No |

## Research Questions

The experiment includes predefined questions that test different types of data reasoning:

| Type | Example Question |
|------|-----------------|
| Aggregation | What is the total revenue from all orders? |
| Lookup | What is the most expensive product in the catalog? |
| Multi-table Join | What products did customer Emily Nakamura purchase? |
| Calculation | What is the average profit margin percentage? |
| Filtering | How many orders are still pending or processing? |
| Analysis | What is the relationship between customer segment and order value? |

## Sample Data Schema

The data follows a star schema pattern common in e-commerce analytics:

```
                    ┌─────────────┐
                    │  suppliers  │
                    └──────┬──────┘
                           │
┌─────────────┐    ┌──────┴──────┐    ┌─────────────┐
│  customers  │────│   orders    │────│ order_items │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                                     ┌───────┴──────┐
                                     │   products   │
                                     └──────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/data/preview` | GET | Preview data in all formats |
| `/api/questions` | GET | Get predefined research questions |
| `/api/experiment/single` | POST | Run single format experiment |
| `/api/experiment/compare` | POST | Run comparison across all formats |

## Results Analysis

For each question, the experiment captures:
- **Answer** - The LLM's response
- **Latency** - Response time in milliseconds
- **Input Tokens** - Size of the prompt
- **Output Tokens** - Size of the response

Compare these metrics across the three formats to draw conclusions about which representation works best for different types of queries.

