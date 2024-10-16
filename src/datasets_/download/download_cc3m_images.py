import os
import PIL.Image
import io
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import urllib
from datasets.utils.file_utils import get_datasets_user_agent
import argparse
import shutil

USER_AGENT = get_datasets_user_agent()

def fetch_single_image(image_url, img_path, idx):
    for _ in range(3):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=5) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
                w, h = image.size
                if w==1 or h==1:
                    return
                path = os.path.join(img_path, f"{idx}.jpg")
                image.save(path, format='JPEG')
            break
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--path_to_index', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.path_to_index):
        raise FileNotFoundError(f"Conceptual Captions 3M index file {args.path_to_index} not found, download it first: "
                                "https://ai.google.com/research/ConceptualCaptions/download")

    path_to_data = os.path.join(args.data_path, "conceptual_captions_3m")
    img_path = os.path.join(path_to_data, "images")
    os.makedirs(img_path, exist_ok=True)

    index_path = os.path.join(path_to_data, "Train-GCC-training.tsv")
    shutil.copy(args.path_to_index, index_path)
    index = pd.read_csv(index_path, sep='\t', header=None).reset_index(drop=True)
    index.columns = ['caption', 'image_url']
    already_existing = os.listdir(img_path)
    already_existing = [int(os.path.splitext(img)[0]) for img in already_existing]
    index = index[~index.index.isin(already_existing)]
    n_workers = os.cpu_count()*4
    with concurrent.futures.ThreadPoolExecutor(n_workers) as executor:
        futures = [executor.submit(fetch_single_image, url, img_path, idx) for idx, url in 
                   tqdm(index['image_url'].items(), total=len(index), desc="Scheduling tasks")]
        list(tqdm(concurrent.futures.as_completed(futures), total=len(index), desc="Downloading images"))

    n_failed = len(index) - len(os.listdir(img_path))
    print(f"Failed to download {n_failed} images (pairs). Percentage: {n_failed/len(index)*100:.2f}%")
    
if __name__ == "__main__":
    main()
