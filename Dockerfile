# Build stage
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Install the project
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source and data
COPY --from=builder /app/src ./src
COPY --from=builder /app/data ./data

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Cloud Run will set PORT environment variable
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "-m", "uvicorn", "src.web_app:app", "--host", "0.0.0.0", "--port", "8080"]

