### Using GPU for TEI

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```


With GPUs
```bash
docker run --rm --gpus all -d nvidia/cuda:11.8.0-base nvidia-smi

docker-compose --profile local-embedding up -d
```

Without GPUs (OpenAI embedding)

```bash
docker-compose up -d
```

### Setup guide
If you plan to not use local embedding, set `LOCAL_EMBEDDING = false` in `.env` file

**Setup for deploy**

- Make `run.sh` file executable
```bash
chmod +x run.sh
```

- For CPU (Using OpenAI Embedding)
```bash
./run.sh --preprocess v3 --openai True 
```

- For GPU (Self-hosted Embedding Server)
```bash
./run.sh local-embedding --preprocess v3 --local True
```

- For GPU + Rerank (Self-hosted Embedding Server)
```bash
./run.sh local-model --preprocess v3 --local True
```

### vLLM

- LLM
```
vllm serve Qwen/Qwen2.5-Coder-3B-Instruct --enable-lora --lora-modules qwen2.5-coder-3b-sft=saves/qwen2.5-3b-coder-test-v3/lora/sft --max-model-len 4096 --max-lora-rank 64 --gpu-memory-utilization 0.8
```

