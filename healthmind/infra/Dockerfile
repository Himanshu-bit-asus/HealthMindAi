# SKELETON, UNTESTED. Pick a CUDA/PyTorch base that matches your pinned versions.
FROM <PYTORCH_CUDA_BASE_IMAGE>
WORKDIR /opt/ml/code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/opt/ml/code
# Tag images by git SHA (healthmind-train:<sha>) so runs are reproducible.
