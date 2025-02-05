### 1. Download the data from Kaggle
Go to GitHub.
Click New Repository.

### 2. Clone the Repository
```
git clone https://github.com/Mayuri14-ctrl/Image-classification.git 
cd your-repository
```
### 3. Add Your Files
Move your data files into the cloned repository folder, then use:
```
git add .
```
#### Commit the changes
```
git commit -m "Added new data files"
git push origin main  # Use 'master' if that's your default branch
```

### 4. Google Colab:
You can easily connect Google Colab to GitHub and run your Python code with GPU support.

Open Google Colab: https://colab.research.google.com/
Select GitHub and choose your repository.
Edit and run your code interactively.

#### Method 1: Open a GitHub Notebook in Colab
If your repository contains Jupyter notebooks (.ipynb files), you can open them directly in Colab:

Go to Google Colab:
👉 https://colab.research.google.com/

Click on the "GitHub" tab.

In the search bar, enter your GitHub username or repo name:

Mayuri14-ctrl/Image-classification
Select the notebook (.ipynb) you want to open and click "Open in Colab".

#### Method 2: Clone a GitHub Repository in Colab
If you want to work with Python scripts (.py) or data files, follow these steps:

### Step 5: Start writing python code 
#### Method 1 Clone the GitHub Repository
Run the following command in a Colab cell:

```
!git clone https://github.com/Mayuri14-ctrl/Image-classification.git
```
This will download the repository into Colab.

##: Change Directory to the Cloned Repo
Navigate into the cloned repository:
```
%cd Image-classification
```
or 
#### Method 2 Clone the Kaggle Repository
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("rishabhrp/chest-x-ray-dataset")

print("Path to dataset files:", path)
!cp -r /root/.cache/kagglehub/datasets/rishabhrp/chest-x-ray-dataset/versions/1 /content/chest-x-ray-dataset
dataset_path = "/content/chest-x-ray-dataset"
```
## Start building model pipeline
### Step 1: Data Preprocessing and Augmentation
In this step, we focus on the essential tasks required to prepare the Chest X-ray dataset for model training. This includes data loading, class balancing, and applying augmentations. Here's a breakdown of the process:

#### 1.1. Loading the Dataset
1.1.1 Setting Dataset Paths:

The dataset's root folder is specified as "./Chest_XRay_Dataset", which contains the images and a CSV file with the ground truth (labels).
The path to the images is stored in images_path, and the path to the CSV file containing the labels is stored in csv_path.
```
dataset_path = "./Chest_XRay_Dataset"
images_path = os.path.join(dataset_path, "xray_images")
csv_path = os.path.join(dataset_path, "Ground_Truth.csv")
```
Loading the CSV File:

The CSV file containing the ground truth labels for each chest X-ray image is loaded into a Pandas DataFrame (df) using pd.read_csv(). The CSV has columns that include the image index (ID) and the labels associated with each image.
```
df = pd.read_csv(csv_path)
```
1.1.2 Filtering Images Based on Availability:

A list of all available image files is retrieved from the xray_images directory using os.listdir().
The DataFrame (df) is then filtered so that it only includes rows where the image index (from the CSV) exists in the images_path directory.
```
image_files = set(os.listdir(images_path))
df = df[df["Image Index"].isin(image_files)]
```
After filtering, the shape of the DataFrame is printed to show the number of rows and columns (i.e., the number of images and their associated labels).
```
print(df.shape)
```
1.1.3 Handling Multi-Label Format:

The Finding Labels column in the CSV contains multiple labels separated by the | character (e.g., "Atelectasis|Effusion").
The lambda function is applied to split these labels into a list for each image.
```
df["Finding Labels"] = df["Finding Labels"].apply(lambda x: x.split("|"))
```
The isnull() function is used to check if there are any missing values, and the first few rows of the DataFrame are printed to verify the changes.
```
print(df.isnull().sum())
print(df.head(2))
```
Resetting the Index:

After processing the labels, the index of the DataFrame is reset using reset_index(). This is useful for organizing the data in a clean format.
```
df = df.reset_index()
```
1.1.4 One-Hot Encoding the Labels:

The multi-label format is then converted into a binary matrix using MultiLabelBinarizer. This means each label (like "Atelectasis", "Effusion", etc.) becomes its own column, and a 1 or 0 is assigned depending on whether the label is present or not for each image.
```
mlb = MultiLabelBinarizer()
df_labels = pd.DataFrame(mlb.fit_transform(df["Finding Labels"]), columns=mlb.classes_)
```
The resulting binary matrix (df_labels) is concatenated with the original DataFrame (df), so now each row has the image index and the corresponding binary labels.
```
df = pd.concat([df, df_labels], axis=1)
```
1.1.5 Dataset Splitting:

The dataset is then split into three parts:
Training Set (80%)
Test Set (20%)
The training set is further split into a validation set (20% of the training set).
This is done using the train_test_split function from sklearn.model_selection.
```
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
```
Saving the Dataset Splits:

Finally, the training, validation, and test DataFrames (train_df, val_df, test_df) are saved as CSV files so they can be used later in training models.
```
train_df.to_csv("train_data.csv", index=False)
val_df.to_csv("val_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
```

#### 1.2 Image Augmentation
Image augmentations are applied to increase the diversity of the training data, which can improve the model's generalization ability. The following augmentations are used:

Gaussian Noise: Adds random noise to images to make the model robust to small perturbations.
```
def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image.astype(np.uint8)
```
Random Rotation: Rotates the image by a random angle between -30 and +30 degrees.

```
def random_rotate(image, max_angle=30):
    angle = random.uniform(-max_angle, max_angle)
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image
```
Random Perspective Transformation: Applies a random perspective transformation to simulate various viewing angles.

```
def random_perspective_transform(image):
    src_pts = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    dst_pts = np.float32([...])  # Randomized destination points
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transformed_image = cv2.warpPerspective(image, matrix, (cols, rows))
    return transformed_image
```
Histogram Equalization: Enhances the contrast of the images by adjusting the intensity distribution.

```
def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
These augmentations are applied in a sequence to each image in the training set.
```
#### 1.3 Preprocessing and Saving Images
preprocess_and_save_images(df, output_folder):

This function takes in a DataFrame (df) and an output folder (output_folder) to process and save images.
For each row in the DataFrame (each image), it reads the image file from the images_path directory, applies a series of transformations, and saves the processed image in a folder structure organized by label.
Steps in this function:
The output folder is created if it doesn't already exist.
For each row, the image file path is determined, and the label is extracted (the first label in the list of labels).
A sub-folder is created for each label to store the processed images.
The image is loaded in grayscale using cv2.imread().
The image is resized to 224x224 pixels (common for deep learning models).
Several preprocessing transformations are applied: Gaussian noise, histogram equalization, random rotation, and perspective transformation.
The processed image is saved into the corresponding label folder.
```
def preprocess_and_save_images(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for _, row in df.iterrows():
        image_path = os.path.join(images_path, row["Image Index"])
        label = row["Finding Labels"][0]  # Assuming first label for simplicity
        
        class_folder = os.path.join(output_folder, label)
        os.makedirs(class_folder, exist_ok=True)  # Create class folder if not exists
        
        if not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = add_gaussian_noise(image)
        image = apply_histogram_equalization(image)
        image = rotate_image(image, angle=np.random.choice([-15, 0, 15]))
        image = perspective_transform(image)
        cv2.imwrite(os.path.join(class_folder, row["Image Index"]), image)
```
Saving Data Splits
Finally, after preprocessing the images and saving them into respective folders, the DataFrame splits (train, validation, and test) are saved as CSV files for future use in training, validation, and testing the model.

```
train_df.to_csv("train_data.csv", index=False)
val_df.to_csv("val_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
```
### Step 2: ORB Feature Extraction for Image Data
In Step 2, the focus shifts to extracting features from the images in the preprocessed dataset using the ORB (Oriented FAST and Rotated BRIEF) method. ORB is a keypoint detection algorithm that is widely used in computer vision, especially for tasks like object recognition or matching, due to its speed and efficiency. Here's a detailed breakdown of this step:

#### 2.1. Set Paths and Initialize ORB
At the start of the script, paths for the preprocessed images and the output CSV file are set. The processed_images_path points to the folder containing the preprocessed training images. The output_csv defines the path where the extracted features will be saved.

##### ORB Initialization:
The cv2.ORB_create() function initializes the ORB feature detector. The argument nfeatures=500 specifies that ORB will detect up to 500 keypoints per image. Keypoints are important locations in the image where distinctive features are detected.
python
Copy
Edit
processed_images_path = "./processed_images/train"  # Path to preprocessed training images
output_csv = "train_orb_features.csv"  # Output CSV file to store extracted features

##### ORB Feature Extractor Initialization
orb = cv2.ORB_create(nfeatures=500)  # Create ORB detector that detects up to 500 keypoints

#### 2.2. Feature Extraction
The script uses the os library to iterate over all class folders in the processed_images_path, and for each class, it processes the images within the folder. For each image, the following operations are carried out:

Load Image:

The image is loaded in grayscale using OpenCV (cv2.IMREAD_GRAYSCALE) because ORB works with grayscale images, which simplifies the detection process.
Detect Keypoints and Compute Descriptors:

The orb.detectAndCompute() function is used to detect keypoints and compute the corresponding descriptors. Descriptors are the unique descriptors that represent the surrounding areas of the detected keypoints.
The keypoints are the points in the image where significant features like corners or edges are detected. The descriptors are vectors that describe the appearance of each keypoint, which can be used for matching or recognition tasks.
Store Features:

The descriptors are stored for each image, and the image file name (Image Index) and class label (Class) are also stored.
Each feature set (image file name, class, descriptors) is saved as a dictionary, and all dictionaries are appended to the features_list.
##### Iterate over class folders
```
for class_name in os.listdir(processed_images_path):
    class_folder = os.path.join(processed_images_path, class_name)

    if not os.path.isdir(class_folder):  # Skip files that are not directories
        continue

    # Iterate over images in each class folder
    for image_file in tqdm(os.listdir(class_folder), desc=f"Processing {class_name}"):
        image_path = os.path.join(class_folder, image_file)

        # Load image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping {image_path}, could not load.")
            continue

        # Detect keypoints and compute descriptors using ORB
        keypoints, descriptors = orb.detectAndCompute(image, None)

        # Store descriptors in a list, converting to a Python list for easy saving
        features_list.append({
            "Image Index": image_file,
            "Class": class_name,
            "Descriptors": descriptors.tolist() if descriptors is not None else None
        })
```
#### 2.3. Save the Extracted Features to CSV
Once the features for all images have been extracted and stored, they are converted into a pandas DataFrame. The DataFrame is then saved as a CSV file (train_orb_features.csv) for later use in machine learning models or other tasks.

Descriptors as Lists:
Since the descriptors are NumPy arrays, they are converted to Python lists using .tolist() to store them in a format compatible with CSV files. If no descriptors are found for a particular image, None is stored instead.
```
# Convert the list of features into a pandas DataFrame
features_df = pd.DataFrame(features_list)

# Save the features to a CSV file
features_df.to_csv(output_csv, index=False)
```
### Step 3: Training a Deep Learning Model for Chest X-Ray Classification
In this step, we train a ResNet-50 deep learning model on the preprocessed X-ray images. This process includes loading data, defining a model, training it, and evaluating its performance.

1️⃣ Preparing the Dataset
We first load the preprocessed images into PyTorch’s DataLoader.

1.1 Define Image Transformations
Neural networks require images in a specific format. We apply:

Conversion to Tensors: Converts images into PyTorch tensors.
Normalization: Matches the mean and standard deviation of ImageNet to help ResNet-50 perform optimally.
python
Copy
Edit
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
1.2 Load the Preprocessed Dataset
Since the images were saved in folders by class (processed_images/train, processed_images/val, processed_images/test), we use datasets.ImageFolder() to read them.

python
Copy
Edit
from torchvision import datasets
from torch.utils.data import DataLoader

train_dataset = datasets.ImageFolder(root='./processed_images/train', transform=transform)
val_dataset = datasets.ImageFolder(root='./processed_images/val', transform=transform)
test_dataset = datasets.ImageFolder(root='./processed_images/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
2️⃣ Defining the Deep Learning Model
We use ResNet-50, a pre-trained CNN model from ImageNet.

2.1 Load Pretrained ResNet-50
python
Copy
Edit
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet50(pretrained=True)
2.2 Freeze Pretrained Layers
We freeze the model’s existing layers so that only the last layer learns from our dataset.

python
Copy
Edit
for param in model.parameters():
    param.requires_grad = False
2.3 Modify the Final Layer
The last layer of ResNet-50 is modified to match the number of classes in our dataset.

python
Copy
Edit
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
2.4 Move Model to GPU (If Available)
We check for a CUDA-enabled GPU and move the model to the device.

python
Copy
Edit
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
3️⃣ Defining the Loss Function & Optimizer
Loss Function: CrossEntropyLoss(), which is used for multi-class classification.
Optimizer: Adam(), applied only to the last layer (since other layers are frozen).
python
Copy
Edit
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
4️⃣ Training the Model
4.1 Define the Training Function
We define a function to train the model for multiple epochs:

python
Copy
Edit
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
4.2 Train the Model
We call the train() function to train the model for 5 epochs.

python
Copy
Edit
train(model, train_loader, criterion, optimizer, device, epochs=5)

 Model Evaluation and Saving
After training the model, we need to evaluate its performance on the validation set. This involves computing accuracy, precision, recall, and F1-score for each class. We use the evaluate() function to do this.

1️⃣ Defining the Evaluation Function
The evaluate() function follows these steps:

Set the model to evaluation mode (model.eval()) so that dropout and batch normalization behave properly.
Disable gradient computation (torch.no_grad()) to improve efficiency.
Loop over the validation set and make predictions.
Store actual labels (y_true) and predicted labels (y_pred).
Ensure only valid classes are included to avoid mismatches in the classification report.
Print a classification report showing precision, recall, and F1-score for each class.
🔹 Function Code Breakdown
python
Copy
Edit
def evaluate(model, val_loader, device):
    model.eval()  # Set model to evaluation mode
    y_true = []  # List to store true labels
    y_pred = []  # List to store predicted labels

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability
            
            y_true.extend(labels.cpu().numpy())  # Convert tensors to numpy arrays
            y_pred.extend(predicted.cpu().numpy())

    # Ensure classes are only those present in both train & validation datasets
    valid_classes = list(sorted(set(train_dataset.classes) & set(val_dataset.classes)))

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=valid_classes))
2️⃣ Running the Evaluation
We call the function to evaluate the model on the validation set.

python
Copy
Edit
evaluate(model, val_loader, device)
📌 Output:

The function prints a classification report, which includes:
Precision: How many predicted cases were actually correct.
Recall: How many actual cases were correctly predicted.
F1-score: The harmonic mean of precision and recall.

✅ Summary of Step 3
Loaded the preprocessed X-ray images using ImageFolder.
Used ResNet-50 as a pre-trained model, freezing all but the last layer.
Modified the final layer to match our dataset’s number of classes.
Defined a loss function (CrossEntropyLoss) and optimizer (Adam).
Trained the model for multiple epochs using GPU acceleration.


