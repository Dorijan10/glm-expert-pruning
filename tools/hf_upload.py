import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import HfApi
api = HfApi()
REPO = 'turintech/GLM-5.2-MaxMin108-GGUF'
api.create_repo(REPO, repo_type='model', private=True, exist_ok=True)
print('repo ready', flush=True)
api.upload_folder(folder_path='/work/hf/maxmin', repo_id=REPO)
print('maxmin done', flush=True)
api.upload_folder(folder_path='/work/hf/fast', repo_id=REPO)
print('UPLOAD DONE', flush=True)
