FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install local packages
RUN pip install -e ./cs336-basics && \
    pip install -e .

# Set default command
CMD ["python", "cs336_systems/benchmarking_script.py"]