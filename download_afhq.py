"""
Download AFHQ dataset from Hugging Face.
AFHQ (Animal Faces-HQ) contains high-quality images of animal faces.
"""

from datasets import load_dataset
import os
from PIL import Image
from tqdm import tqdm


def download_afhq(output_dir='./data/afhq'):
    """
    Download and organize AFHQ dataset from Hugging Face.

    Args:
        output_dir: Directory to save the dataset
    """
    print("Downloading AFHQ dataset from Hugging Face...")

    # Load dataset from Hugging Face
    # AFHQ is available at https://huggingface.co/datasets/huggan/AFHQv2
    dataset = load_dataset("huggan/AFHQv2", trust_remote_code=True)

    print(f"\nDataset loaded successfully!")
    print(f"Splits: {list(dataset.keys())}")

    # Process each split
    for split_name in dataset.keys():
        print(f"\nProcessing {split_name} split...")
        split_data = dataset[split_name]

        # AFHQ has 'image' and 'label' (0=cat, 1=dog, 2=wild)
        label_names = {0: 'cat', 1: 'dog', 2: 'wild'}

        # Create directories
        for label_name in label_names.values():
            split_dir = os.path.join(output_dir, split_name, label_name)
            os.makedirs(split_dir, exist_ok=True)

        # Save images
        for idx, item in enumerate(tqdm(split_data, desc=f"Saving {split_name} images")):
            image = item['image']
            label = item['label']
            label_name = label_names[label]

            # Save image
            image_path = os.path.join(
                output_dir,
                split_name,
                label_name,
                f'{idx:05d}.jpg'
            )

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image.save(image_path, quality=95)

    print(f"\n✓ Dataset saved to {output_dir}")
    print("\nDataset structure:")
    for split_name in dataset.keys():
        print(f"\n{split_name}/")
        for label_name in ['cat', 'dog', 'wild']:
            split_dir = os.path.join(output_dir, split_name, label_name)
            count = len(os.listdir(split_dir)) if os.path.exists(split_dir) else 0
            print(f"  {label_name}/: {count} images")


if __name__ == "__main__":
    download_afhq()
