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
./run.sh --openai True
```

- For GPU (Self-hosted Embedding Server)
```bash
./run.sh local-embedding --local True
```