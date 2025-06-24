import json
import os
import requests
from tqdm import tqdm

def download_nocaps_images(image_info_json_path, output_image_dir):
    """
    Downloads NoCaps images from URLs specified in the image info JSON.
    """
    print(f"Loading image info from: {image_info_json_path}")
    if not os.path.exists(image_info_json_path):
        print(f"Error: Image info JSON not found at {image_info_json_path}")
        return

    with open(image_info_json_path, 'r') as f:
        image_info = json.load(f)

    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Images will be downloaded to: {output_image_dir}")

    download_count = 0
    skipped_count = 0
    error_count = 0

    # The JSON structure has an 'images' key, containing a list of image dicts
    images_list = image_info.get('images', [])

    for img_data in tqdm(images_list, desc="Downloading NoCaps Images"):
        image_id = img_data.get('id')
        coco_url = img_data.get('coco_url')
        file_name = img_data.get('file_name')

        if not all([coco_url, file_name]):
            print(f"Warning: Missing URL or filename for image ID {image_id}. Skipping.")
            error_count += 1
            continue

        image_path = os.path.join(output_image_dir, file_name)

        if os.path.exists(image_path):
            skipped_count += 1
            continue # Skip if already downloaded

        try:
            response = requests.get(coco_url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            download_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {coco_url} to {image_path}: {e}")
            error_count += 1
        except Exception as e:
            print(f"An unexpected error occurred for {coco_url}: {e}")
            error_count += 1

    print(f"\nDownload complete.")
    print(f"Successfully downloaded: {download_count} images.")
    print(f"Skipped (already existing): {skipped_count} images.")
    print(f"Errors during download: {error_count} images.")


if __name__ == "__main__":
    # Define paths consistent with your project structure
    # You might need to adjust 'image_info_json_path' and 'output_image_dir'
    
    # Assuming you saved the downloaded JSON here:
    image_info_json_path = "./datasets/nocaps/nocaps_test_public.json" 
    
    # This should be where your data_loader expects images:
    output_image_dir = "./datasets/nocaps/images" 

    download_nocaps_images(image_info_json_path, output_image_dir)
