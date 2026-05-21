import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class EchoDataset(Dataset):
    def __init__(self, root, view='A4C', split='train', T=32, img_size=112, augment=False):
        self.root = root
        self.view = view
        self.T = T
        self.img_size = img_size
        self.augment = augment

        csv_path = os.path.join(root, view, 'FileList.csv')
        df = pd.read_csv(csv_path)

        # Normalize column names (strip whitespace, handle case)
        df.columns = df.columns.str.strip()

        if split == 'train':
            self.df = df[df['Split'].isin(range(8))].reset_index(drop=True)
        elif split == 'val':
            self.df = df[df['Split'] == 8].reset_index(drop=True)
        elif split == 'test':
            self.df = df[df['Split'] == 9].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.video_dir = os.path.join(root, view, 'Videos')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['FileName']
        ef = float(row['EF'])

        video_path = os.path.join(self.video_dir, filename)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            frames = torch.zeros(self.T, 1, self.img_size, self.img_size)
            return frames, torch.tensor(ef, dtype=torch.float32), filename, self.T

        indices = np.linspace(0, total_frames - 1, self.T, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)
        cap.release()

        # (T, H, W) -> (T, 1, H, W), float32, [0, 1]
        frames = np.stack(frames, axis=0).astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).unsqueeze(1)

        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                frames = torch.flip(frames, dims=[-1])

            # Brightness jitter ±10%
            factor = 1.0 + (torch.rand(1).item() * 0.2 - 0.1)
            frames = torch.clamp(frames * factor, 0.0, 1.0)

            # Random temporal crop: 28 of 32 frames
            T_crop = 28
            if frames.shape[0] > T_crop:
                start = torch.randint(0, frames.shape[0] - T_crop + 1, (1,)).item()
                frames = frames[start:start + T_crop]

        length = frames.shape[0]
        return frames, torch.tensor(ef, dtype=torch.float32), filename, length


def pad_collate(batch):
    frames_list, ef_list, names, lengths = zip(*batch)

    # Sort descending by length (required for pack_padded_sequence)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    frames_list = [frames_list[i] for i in order]
    ef_list     = [ef_list[i]     for i in order]
    names       = [names[i]       for i in order]
    lengths     = [lengths[i]     for i in order]

    # pad_sequence expects list of (T, 1, H, W) -> output (B, T_max, 1, H, W)
    frames_padded   = pad_sequence(frames_list, batch_first=True, padding_value=0.0)
    ef_tensor       = torch.stack(ef_list)
    lengths_tensor  = torch.tensor(lengths)

    return frames_padded, ef_tensor, names, lengths_tensor


def parse_volume_tracings(csv_path):
    """
    Returns dict: {filename: {ed_frame, es_frame, contour_ed, contour_es}}
    ED = larger ventricular area (max filling), ES = smaller (max contraction).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    result = {}

    for filename, group in df.groupby('FileName'):
        frame_ids = group['Frame'].unique()
        if len(frame_ids) < 2:
            continue

        areas = {}
        for fid in frame_ids:
            pts = group[group['Frame'] == fid][['X', 'Y']].values
            if len(pts) >= 3:
                x, y = pts[:, 0], pts[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            else:
                area = 0.0
            areas[fid] = area

        ed_frame = max(areas, key=areas.get)
        es_frame = min(areas, key=areas.get)

        result[filename] = {
            'ed_frame':   int(ed_frame),
            'es_frame':   int(es_frame),
            'contour_ed': group[group['Frame'] == ed_frame][['X', 'Y']].values.tolist(),
            'contour_es': group[group['Frame'] == es_frame][['X', 'Y']].values.tolist(),
        }

    return result
