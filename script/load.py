from modules.pvcnn import PVConv # or the correct model class from PVConv repo

# Example parameters (adjust classes and channels)
num_classes = 13  # e.g. S3DIS semantic classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PVConv(num_classes=num_classes).to(device)

# Load checkpoint
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

