
import os
import csv

class LabelCollector:
    def __init__(self, image_root="all_images", output_csv="labels.csv"):
        self.image_root = image_root
        self.output_csv = output_csv
        self.valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        self.rows = []

    def scan_images(self):
        print("🔍 Scanning for images...")
        for root, _, files in os.walk(self.image_root):
            for fname in files:
                if fname.lower().endswith(self.valid_exts):
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, self.image_root)
                    self._collect_label(rel_path)

    def _collect_label(self, rel_path):
        while True:
            label = input(f"Image: {rel_path}\nLabel [perforated / non_perforated]: ").strip().lower()
            if label in ['perforated', 'non_perforated']:
                self.rows.append([rel_path, label])
                break
            else:
                print(" Invalid label. Please enter 'perforated' or 'non_perforated'.")

    def save_csv(self):
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(self.rows)
        print(f"\n CSV created: {self.output_csv} ({len(self.rows)} labeled images)")

    def run(self):
        self.scan_images()
        self.save_csv()
