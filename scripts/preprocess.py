# scripts/preprocess.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(image_path, img_size):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img / 255.0
    return img

def load_dataset(extract_paths, img_size, max_total_files=15000):
    X, y = [], []
    total_files, unreadable_files, valid_files = 0, 0, 0

    class_file_counts = {}
    remaining_files = max_total_files

    classes = [os.path.basename(path) for path in extract_paths]
    print(f'Found classes: {classes}')

    for class_idx, (class_name, class_path) in enumerate(zip(classes, extract_paths)):
        print(f'Processing class: {class_name}')
        files = os.listdir(class_path)
        total_files += len(files)
        class_file_counts[class_name] = {'total': len(files), 'valid': 0, 'unreadable': 0}

        max_files_per_class = min(len(files), remaining_files // (len(classes) - class_idx))
        files = files[:max_files_per_class]
        remaining_files -= max_files_per_class

        for img_name in files:
            img_path = os.path.join(class_path, img_name)
            if not any(img_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']):
                continue
            img = load_and_preprocess_image(img_path, img_size)
            if img is not None:
                X.append(img)
                y.append(class_idx)
                valid_files += 1
                class_file_counts[class_name]['valid'] += 1
            else:
                print(f'Warning: {img_path} cannot be read.')
                unreadable_files += 1
                class_file_counts[class_name]['unreadable'] += 1

    print(f'Total files: {total_files}')
    print(f'Valid files: {valid_files}')
    print(f'Unreadable files: {unreadable_files}')

    for class_name, counts in class_file_counts.items():
        print(f"{class_name}: Total = {counts['total']}, Valid = {counts['valid']}, Unreadable = {counts['unreadable']}")

    return np.array(X), np.array(y)

def run(input_dir, output_dir):
    extract_paths = [
        os.path.join(input_dir, 'Normal'),
        os.path.join(input_dir, 'Lung-Opacity'),
        os.path.join(input_dir, 'COVID-19'),
        os.path.join(input_dir, 'Tuberculosis'),
        os.path.join(input_dir, 'Pneumonia')
    ]
    img_size = (256, 256)
    X, y = load_dataset(extract_paths, img_size)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    np.savez(os.path.join(output_dir, 'train.npz'), X_train=X_train, y_train=y_train)
    np.savez(os.path.join(output_dir, 'val.npz'), X_val=X_val, y_val=y_val)
    np.savez(os.path.join(output_dir, 'test.npz'), X_test=X_test, y_test=y_test)
