from huggingface_hub import login, snapshot_download

import argparse
import os
import sys

def get_args():
    
    paser = argparse.ArgumentParser()
    paser.add_argument('--repo_id', type=str)
    
    return paser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    
    hf_api_token = os.getenv('HF_API_TOKEN')

    repo_id = args.repo_id

    save_path = os.path.join(os.getcwd(), 'saves', repo_id)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    login(token=hf_api_token)

    snapshot_download(local_dir=save_path, repo_id=repo_id, repo_type="model")