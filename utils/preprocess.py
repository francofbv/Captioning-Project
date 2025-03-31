import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class UCF101Dataset(Dataset):
    """
    Dataset class for UCF101 videos
    """
    def __init__(self, 
                 root_dir,
                 annotation_file,
                 num_frames=16,          # Number of frames to sample from each video
                 frame_size=224,         # Size to resize frames to
                 mode='train',
                 transform=None):
        """
        Args:
            root_dir (str): Directory with all the video files
            annotation_file (str): Path to annotation file with video paths and labels
            num_frames (int): Number of frames to sample from each video
            frame_size (int): Size to resize frames to
            mode (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on frames
        """
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.mode = mode
        
        # Read annotation file
        with open(annotation_file, 'r') as f:
            self.video_paths = [line.strip().split()[0] for line in f]
            self.labels = [int(line.strip().split()[1]) for line in f]
            
        # Define default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((frame_size, frame_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def load_video(self, video_path):
        """Load video and sample frames"""
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        # Get total frames in video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"No frames found in video: {video_path}")
            
        # Calculate sampling rate to get desired number of frames
        sample_rate = max(total_frames // self.num_frames, 1)
        
        frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
        cap.release()
        
        if len(frames) < self.num_frames:
            # If we couldn't get enough frames, duplicate the last frame
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
                
        return frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        Returns:
            frames_tensor (torch.Tensor): Tensor of shape [num_frames, channels, height, width]
            label (int): Class label
        """
        video_path = self.root_dir / self.video_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess frames
        try:
            frames = self.load_video(video_path)
            
            # Apply transforms to each frame
            frames_tensor = torch.stack([self.transform(frame) for frame in frames])
            
            return frames_tensor, label
            
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            # Return a zero tensor and the label if there's an error
            return torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size), label

def create_data_loaders(root_dir, 
                       train_annotation_file,
                       test_annotation_file,
                       batch_size=32,
                       num_frames=16,
                       frame_size=224,
                       num_workers=4):
    """
    Create train and test data loaders
    """
    train_dataset = UCF101Dataset(
        root_dir=root_dir,
        annotation_file=train_annotation_file,
        num_frames=num_frames,
        frame_size=frame_size,
        mode='train'
    )
    
    test_dataset = UCF101Dataset(
        root_dir=root_dir,
        annotation_file=test_annotation_file,
        num_frames=num_frames,
        frame_size=frame_size,
        mode='test'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    root_dir = "path/to/UCF101/videos"
    train_annotation_file = "path/to/trainlist01.txt"
    test_annotation_file = "path/to/testlist01.txt"
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        root_dir=root_dir,
        train_annotation_file=train_annotation_file,
        test_annotation_file=test_annotation_file,
        batch_size=32,
        num_frames=16,
        frame_size=224
    )
    
    # Test the data loader
    for batch_idx, (frames, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}")
        print(f"Frames shape: {frames.shape}")  # Should be [batch_size, num_frames, channels, height, width]
        print(f"Labels shape: {labels.shape}")  # Should be [batch_size]
        break