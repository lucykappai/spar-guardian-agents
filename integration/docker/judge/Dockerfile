# CheatBench's judge imports Linux-only sandbox code (core/sandbox/runtime.py dlopens
# libc.so.6 at import), so judge.py runs in this container on macOS hosts.
# Build:  docker build -t cheatbench-judge integration/docker/judge
# Run:    docker run --rm -v "$ROOT:$ROOT" -w "$ROOT/submodules/cheatbench" -e OPENAI_API_KEY \
#             cheatbench-judge python judge.py "$ROOT/runs/<run>" --judge gpt-6-sol \
#             --models-config "$ROOT/integration/task/models.yaml"
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "litellm>=1.0" "openai>=1.0" "anthropic>=0.121.0" "pyyaml>=6.0" \
        "python-dotenv>=1.0" "requests>=2.31" "tqdm>=4.66" "pydantic>=2.0" "chess>=1.11.2"
