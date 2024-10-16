from huggingface_hub import snapshot_download
import argparse
import os
import tarfile
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    path_to_data = os.path.join(args.data_path, "imagenet")
    path_to_train_split = os.path.join(path_to_data, "train")
    path_to_val_split = os.path.join(path_to_data, "val")
    os.makedirs(path_to_train_split, exist_ok=True)
    os.makedirs(path_to_val_split, exist_ok=True)
    
    snapshot_download(repo_id="ILSVRC/imagenet-1k", repo_type="dataset",
                      allow_patterns=["data/train_images_*", "data/val_images_*"], local_dir=path_to_data)

    pattern = os.path.join(path_to_data, 'data', 'train_images*.tar.gz')
    train_archives = glob.glob(pattern)
    print(f"Extracting {len(train_archives)} train archives")
    for file in train_archives:
        with tarfile.open(file, 'r') as tar:
            tar.extractall(path_to_train_split)
        os.remove(file)

    val_archive = os.path.join(path_to_data, 'data', 'val_images.tar.gz')
    print("Extracting val archive")
    with tarfile.open(val_archive, 'r') as tar:
        tar.extractall(path_to_val_split)
    os.remove(val_archive)

if __name__ == "__main__":
    main()
