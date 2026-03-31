"""
Dataset classes for GLA-FatigueNet.
Supports FER2013, fatigue-specific datasets, and image folder formats.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path

from data.landmark_extractor import LandmarkExtractor
from data.augmentation import get_train_transforms, get_val_transforms


class FatigueEmotionDataset(Dataset):
    """
    Multi-task dataset for simultaneous fatigue and emotion detection.
    
    Supports two data formats:
    1. Image folder: data_root/{split}/{class_name}/image.jpg
    2. FER2013 CSV: pixels column with 48x48 grayscale images
    
    Each sample returns:
        - image: transformed tensor (3, H, W)
        - geometric_features: (15,) float tensor
        - emotion_label: int (0-6)
        - fatigue_label: int (0-2)
    """

    def __init__(self, config, split='train', transform=None,
                 precomputed_landmarks=None):
        """
        Args:
            config: dict from config.yaml
            split: 'train', 'val', or 'test'
            transform: albumentations transform pipeline
            precomputed_landmarks: optional dict mapping image_path -> features
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.precomputed_landmarks = precomputed_landmarks or {}
        self.image_size = config['data']['image_size']
        
        self.emotion_classes = config['data']['emotion_classes']
        self.fatigue_classes = config['data']['fatigue_classes']
        self.num_emotion_classes = len(self.emotion_classes)
        self.num_fatigue_classes = len(self.fatigue_classes)
        self.num_geometric_features = config['model']['gla']['geometric_features']
        
        # Initialize landmark extractor
        self.landmark_extractor = LandmarkExtractor(
            num_features=self.num_geometric_features
        )
        
        # Load data
        self.samples = []  # List of (image_path_or_pixels, emotion_label)
        self.data_format = self._detect_format(config)
        
        if self.data_format == 'folder':
            self._load_folder_data(config, split)
        elif self.data_format == 'csv':
            self._load_csv_data(config, split)
        elif self.data_format == 'synthetic':
            self._generate_synthetic_data(config)
        
        print(f"[INFO] {split} dataset: {len(self.samples)} samples "
              f"(format: {self.data_format})")

    def _detect_format(self, config):
        """Detect dataset format based on available files."""
        data_root = config['data']['data_root']
        
        # Check for image folder structure
        if os.path.isdir(data_root):
            train_dir = os.path.join(data_root, 'train')
            if os.path.isdir(train_dir):
                return 'folder'
        
        # Check for FER2013 CSV
        csv_path = os.path.join(data_root, 'fer2013.csv')
        if os.path.isfile(csv_path):
            return 'csv'
        
        # Check for fatigue dataset
        fatigue_root = config['data'].get('fatigue_data_root', '')
        if os.path.isdir(fatigue_root):
            return 'folder'
        
        # Fall back to synthetic for testing
        print("[WARNING] No dataset found. Using synthetic data for testing.")
        return 'synthetic'

    def _load_folder_data(self, config, split):
        """Load data from image folder structure."""
        data_root = config['data']['data_root']
        fatigue_root = config['data'].get('fatigue_data_root', '')
        
        split_dir = os.path.join(data_root, split)
        
        if not os.path.isdir(split_dir):
            # Try fatigue dataset
            split_dir = os.path.join(fatigue_root, split)
        
        if not os.path.isdir(split_dir):
            print(f"[WARNING] Split directory not found: {split_dir}")
            self._generate_synthetic_data(config)
            return

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for class_idx, class_name in enumerate(sorted(os.listdir(split_dir))):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # Map class name to emotion label
            emotion_label = self._get_emotion_label(class_name)
            
            for img_name in os.listdir(class_dir):
                if Path(img_name).suffix.lower() in valid_extensions:
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append({
                        'path': img_path,
                        'emotion_label': emotion_label,
                        'format': 'file'
                    })

    def _load_csv_data(self, config, split):
        """Load data from FER2013 CSV format."""
        csv_path = os.path.join(config['data']['data_root'], 'fer2013.csv')
        
        if not os.path.isfile(csv_path):
            print(f"[WARNING] CSV file not found: {csv_path}")
            self._generate_synthetic_data(config)
            return
        
        df = pd.read_csv(csv_path)
        
        # FER2013 Usage column: 0=Training, 1=PublicTest, 2=PrivateTest
        split_map = {'train': 'Training', 'val': 'PublicTest', 'test': 'PrivateTest'}
        usage = split_map.get(split, 'Training')
        
        df_split = df[df['Usage'] == usage]
        
        for _, row in df_split.iterrows():
            pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
            self.samples.append({
                'pixels': pixels,
                'emotion_label': int(row['emotion']),
                'format': 'pixels'
            })

    def _generate_synthetic_data(self, config, num_samples=500):
        """Generate synthetic data for testing the pipeline."""
        self.data_format = 'synthetic'
        
        for i in range(num_samples):
            emotion_label = np.random.randint(0, self.num_emotion_classes)
            self.samples.append({
                'emotion_label': emotion_label,
                'format': 'synthetic'
            })

    def _get_emotion_label(self, class_name):
        """Map class name to emotion label index."""
        class_name_lower = class_name.lower().strip()
        
        for idx, name in enumerate(self.emotion_classes):
            if class_name_lower == name.lower() or class_name_lower.startswith(name.lower()):
                return idx
        
        # Try numeric class names
        try:
            return int(class_name)
        except ValueError:
            pass
        
        # Fatigue-specific mappings
        fatigue_map = {
            'alert': 6,       # Map to neutral
            'drowsy': 4,      # Map to sad
            'fatigued': 2,    # Map to fear
            'awake': 6,
            'sleepy': 4,
            'yawning': 5,     # Map to surprise (mouth open)
        }
        
        return fatigue_map.get(class_name_lower, 6)  # Default: neutral

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        emotion_label = sample['emotion_label']

        # Load image
        if sample['format'] == 'file':
            image = cv2.imread(sample['path'])
            if image is None:
                # Fallback to synthetic
                image = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif sample['format'] == 'pixels':
            # FER2013: 48x48 grayscale → resize and convert to RGB
            gray = sample['pixels']
            image = cv2.resize(gray, (self.image_size, self.image_size))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Synthetic
            image = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)

        # Extract geometric features (or use precomputed)
        cache_key = sample.get('path', f'synthetic_{idx}')
        if cache_key in self.precomputed_landmarks:
            geometric_features = self.precomputed_landmarks[cache_key]
        else:
            try:
                geometric_features = self.landmark_extractor.extract_features(image)
            except Exception:
                geometric_features = np.zeros(self.num_geometric_features, dtype=np.float32)

        # Derive fatigue label
        fatigue_label = self.landmark_extractor.get_fatigue_label(
            geometric_features, emotion_label
        )

        # Apply data augmentation
        if self.transform:
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
        else:
            # Basic fallback transform
            image = cv2.resize(image, (self.image_size, self.image_size))
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        geometric_tensor = torch.from_numpy(geometric_features).float()
        emotion_tensor = torch.tensor(emotion_label, dtype=torch.long)
        fatigue_tensor = torch.tensor(fatigue_label, dtype=torch.long)

        return {
            'image': image_tensor,
            'geometric_features': geometric_tensor,
            'emotion_label': emotion_tensor,
            'fatigue_label': fatigue_tensor,
        }


def create_dataloaders(config):
    """Create train, validation, and test data loaders."""
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    train_dataset = FatigueEmotionDataset(config, split='train', transform=train_transform)
    val_dataset = FatigueEmotionDataset(config, split='val', transform=val_transform)
    test_dataset = FatigueEmotionDataset(config, split='test', transform=val_transform)

    batch_size = config['training']['batch_size']
    num_workers = config['data']['num_workers']
    pin_memory = config['data']['pin_memory']

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
