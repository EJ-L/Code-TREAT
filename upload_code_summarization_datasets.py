#!/usr/bin/env python3
"""
Script to upload code summarization datasets to Hugging Face repositories
"""
from huggingface_hub import HfApi
from pathlib import Path
import pandas as pd

def get_file_info(file_path):
    """Get basic information about the file"""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"📁 File: {path.name}")
    print(f"📊 Size: {size_mb:.2f} MB")
    
    # Try to count lines (for rough record estimate)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f if line.strip())
        print(f"📝 Approximate records: {line_count}")
    except Exception as e:
        print(f"⚠️ Could not count records: {e}")
    
    return {"path": path, "size_mb": size_mb}

def convert_to_parquet(jsonl_path, output_dir, filename_prefix):
    """Convert JSONL to parquet format"""
    import pandas as pd
    from pathlib import Path
    
    print(f"🔄 Converting {jsonl_path} to parquet...")
    
    # Read and filter empty lines
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    
    # Convert to DataFrame
    df = pd.read_json("\n".join(lines), lines=True)
    
    # Generate parquet filename
    parquet_path = Path(output_dir) / f"{filename_prefix}.parquet"
    
    # Save as parquet
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    
    print(f"✅ Parquet saved: {parquet_path}")
    return str(parquet_path)

def upload_to_huggingface(file_path, repo_id, filename_prefix):
    """Upload JSONL and parquet files to Hugging Face dataset repository"""
    
    # Get file info
    file_info = get_file_info(file_path)
    if not file_info:
        return False
    
    try:
        print(f"\n🚀 Uploading to {repo_id}...")
        
        api = HfApi()
        
        # Convert to parquet first
        output_dir = Path(file_path).parent
        parquet_path = convert_to_parquet(file_path, output_dir, filename_prefix)
        
        # Upload JSONL to raw directory
        jsonl_target = f"{filename_prefix}.jsonl"
        print(f"📤 Uploading JSONL as {jsonl_target}...")
        
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"./raw/{jsonl_target}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {jsonl_target} dataset"
        )
        
        print(f"✅ Successfully uploaded JSONL to {repo_id}/raw/{jsonl_target}")
        
        # Upload parquet to data directory
        parquet_target = f"{filename_prefix}.parquet"
        print(f"📤 Uploading parquet as {parquet_target}...")
        
        api.upload_file(
            path_or_fileobj=parquet_path,
            path_in_repo=f"./data/{parquet_target}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {parquet_target} dataset"
        )
        
        print(f"✅ Successfully uploaded parquet to {repo_id}/data/{parquet_target}")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    """Main function to upload both datasets"""
    print("🎯 Uploading Code Summarization Datasets to Hugging Face")
    print("=" * 50)
    
    # Dataset configurations
    datasets = [
        {
            "file_path": "/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/code_summarization/parsed_functions/merged_functions.jsonl",
            "repo_id": "Code-TREAT/code_summarization",
            "filename_prefix": "code_summarization_gh_2023",
            "description": "Full GitHub 2023 Code Summarization Dataset"
        },
        {
            "file_path": "/Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/code_summarization/parsed_functions/merged_functions_lite.jsonl", 
            "repo_id": "Code-TREAT/code_summarization_lite",
            "filename_prefix": "code_summarization_gh_2023",
            "description": "Lite GitHub 2023 Code Summarization Dataset"
        }
    ]
    
    success_count = 0
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n📋 Dataset {i}/2: {dataset['description']}")
        print("-" * 40)
        
        success = upload_to_huggingface(
            file_path=dataset["file_path"],
            repo_id=dataset["repo_id"],
            filename_prefix=dataset["filename_prefix"]
        )
        
        if success:
            success_count += 1
        
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Upload Summary: {success_count}/{len(datasets)} successful")
    
    if success_count == len(datasets):
        print("🎉 All datasets uploaded successfully!")
    elif success_count > 0:
        print("⚠️ Some datasets uploaded successfully, check errors above")
    else:
        print("❌ All uploads failed, check your HuggingFace credentials")
    
    print("\n📖 Uploaded datasets:")
    for dataset in datasets:
        print(f"  • {dataset['repo_id']}")
    
if __name__ == "__main__":
    main()