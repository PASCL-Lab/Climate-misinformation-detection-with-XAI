from huggingface_hub import HfApi, upload_folder

repo_id = "ImaanIbrar/Climate-misinformation-classification"

# 3. Path to your local model folder (the one with config.json, pytorch_model.bin, etc.)
local_folder = "backend/model/final_model_augv2_89"

# 4. Upload
upload_folder(
    repo_id=repo_id,
    folder_path=local_folder,
    commit_message="Initial upload of model"
)

print(f"✅ Model uploaded successfully! Check it here: https://huggingface.co/{repo_id}")
