import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from torchvision import models
import timm
from facenet_pytorch import InceptionResnetV1, MTCNN

# Advanced loss function - Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# Squeeze-and-Excitation block to enhance feature attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3], keepdim=True)
        se = self.fc1(se.squeeze(-1).squeeze(-1))
        se = nn.ReLU()(se)
        se = self.fc2(se)
        se = self.sigmoid(se).unsqueeze(-1).unsqueeze(-1)
        return x * se

# EfficientNet with SEBlock for feature enhancement
class EfficientNetWithSE(nn.Module):
    def __init__(self):
        super(EfficientNetWithSE, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        num_features = self.efficientnet.classifier[1].in_features
        self.se = SEBlock(num_features)  # Adding SE block for attention
        self.efficientnet.classifier[1] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.efficientnet.features(x)
        x = self.se(x)  # Apply SE block
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.efficientnet.classifier(x)
        return x

# LSTM for temporal sequence learning across frames
class LSTMSequence(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=2):
        super(LSTMSequence, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(lstm_out)

# MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Preprocessing: Detect and crop face before feeding to models
def preprocess_frames(frames):
    processed_frames = []
    for frame in frames:
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                # Crop the face region from the frame
                face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                face = preprocess_image(face)  # Apply transforms on the cropped face
                processed_frames.append(face)
    return processed_frames

# General preprocessing for each face frame
def preprocess_image(frame):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(frame)

# Function to extract frames from a video
def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    
    while success:
        frames.append(frame)
        success, frame = video.read()
    
    video.release()
    return frames

# Predict fake/real using the ensemble model
def predict_deepfake(frames, model, step=5):
    total_fake_prob = 0
    count = 0
    processed_frames = preprocess_frames(frames)
    
    for i in range(0, len(processed_frames), step):
        input_tensor = processed_frames[i].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        fake_prob = output.item()
        total_fake_prob += fake_prob
        count += 1

    average_fake_prob = total_fake_prob / count
    return average_fake_prob

# Ensemble with EfficientNet, XceptionNet, ResNet50, and InceptionResNetV1
class ModelEnsembleWithLSTM(nn.Module):
    def __init__(self, models_list, lstm_model, weights=None):
        super(ModelEnsembleWithLSTM, self).__init__()
        self.models = models_list
        self.lstm_model = lstm_model  # LSTM for temporal learning
        self.num_models = len(models_list)
        if weights is None:
            self.weights = [1 / self.num_models] * self.num_models
        else:
            self.weights = weights

    def forward(self, x):
        outputs = []
        for i, model in enumerate(self.models):
            outputs.append(model(x))
        combined_output = torch.cat(outputs, dim=1)
        return self.lstm_model(combined_output)

class ResNet50Model(nn.Module):
    def __init__(self):
        super(ResNet50Model, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)

class XceptionNetModel(nn.Module):
    def __init__(self):
        super(XceptionNetModel, self).__init__()
        self.xception = timm.create_model('xception', pretrained=True)
        num_features = self.xception.get_classifier().in_features
        self.xception.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.xception(x)

from facenet_pytorch import InceptionResnetV1

class InceptionResNetModel(nn.Module):
    def __init__(self):
        super(InceptionResNetModel, self).__init__()
        # Load pre-trained InceptionResnetV1 model (commonly used for face recognition)
        self.inception_resnet = InceptionResnetV1(pretrained='vggface2')
        # Freeze layers of the pre-trained model (optional, for transfer learning)
        for param in self.inception_resnet.parameters():
            param.requires_grad = False
        # Adding custom fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Get the feature embeddings from InceptionResnetV1
        x = self.inception_resnet(x)
        # Pass through the custom classifier
        x = self.fc(x)
        return x

class LSTMSequence(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=2):
        super(LSTMSequence, self).__init__()
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # Fully connected layer for final classification
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional LSTM
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass input through LSTM
        lstm_out, _ = self.lstm(x)
        # Take the last output from the LSTM and pass it through FC layer
        lstm_out = self.fc(lstm_out[:, -1, :])
        # Apply Sigmoid for binary classification
        return self.sigmoid(lstm_out)

class InceptionResNetModel(nn.Module):
    def __init__(self):
        super(InceptionResNetModel, self).__init__()
        # Load pre-trained InceptionResnetV1 model (commonly used for face recognition)
        self.inception_resnet = InceptionResnetV1(pretrained='vggface2')
        # Freeze layers of the pre-trained model (optional, for transfer learning)
        for param in self.inception_resnet.parameters():
            param.requires_grad = False
        # Adding custom fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Ensure model is in evaluation mode
        self.inception_resnet.eval()
        # Get the feature embeddings from InceptionResnetV1
        x = self.inception_resnet(x)
        # Pass through the custom classifier
        x = self.fc(x)
        return x

class LSTMSequence(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=2):
        super(LSTMSequence, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional LSTM
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        print(f"Dimensions of LSTM output: {lstm_out.shape}")  # Debugging line
        if len(lstm_out.shape) == 3:  # Check if it's 3D
            lstm_out = self.fc(lstm_out[:, -1, :])
        elif len(lstm_out.shape) == 2:  # If it's 2D, assume sequence length of 1
            lstm_out = self.fc(lstm_out[:, :])
        else:
            raise ValueError("Unexpected LSTM output dimensions")
        return self.sigmoid(lstm_out)


# Initialize individual models
resnet_model = ResNet50Model()
xception_model = XceptionNetModel()
efficientnet_model = EfficientNetWithSE()
inception_resnet_model = InceptionResNetModel()
lstm_model = LSTMSequence(input_size=4)  # Input size = number of models

# Ensemble model with LSTM for temporal learning
ensemble_model = ModelEnsembleWithLSTM(
    [resnet_model, xception_model, efficientnet_model, inception_resnet_model],
    lstm_model, weights=[0.15, 0.35, 0.25, 0.25]
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ensemble_model.to(device)
ensemble_model.eval()

# Main function to run the deepfake detection pipeline
def deepfake_detection():
    video_path = input("Enter the path")
    print("Extracting frames from the video...")
    frames = extract_frames(video_path)
    print("Running deepfake detection on frames...")
    fake_prob = predict_deepfake(frames, ensemble_model)
    classify_video(fake_prob)

# Classify the video based on prediction probability
def classify_video(fake_prob):
    if fake_prob > 0.5:
        print(f"The video is FAKE with {fake_prob * 100:.2f}% confidence.")
    else:
        print(f"The video is REAL with {(1 - fake_prob) * 100:.2f}% confidence.")

# Run the deepfake detection
#deepfake_detection()
