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
# Commit the changes
```
git commit -m "Added new data files"
git push origin main  # Use 'master' if that's your default branch
```

## 4. Google Colab:
You can easily connect Google Colab to GitHub and run your Python code with GPU support.

Open Google Colab: https://colab.research.google.com/
Select GitHub and choose your repository.
Edit and run your code interactively.

### Method 1: Open a GitHub Notebook in Colab
If your repository contains Jupyter notebooks (.ipynb files), you can open them directly in Colab:

Go to Google Colab:
👉 https://colab.research.google.com/

Click on the "GitHub" tab.

In the search bar, enter your GitHub username or repo name:

Mayuri14-ctrl/Image-classification
Select the notebook (.ipynb) you want to open and click "Open in Colab".

### Method 2: Clone a GitHub Repository in Colab
If you want to work with Python scripts (.py) or data files, follow these steps:

### Step 5: Start writing python code 
# Method 1 Clone the GitHub Repository
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
# Method 2 Clone the Kaggle Repository
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("rishabhrp/chest-x-ray-dataset")

print("Path to dataset files:", path)
!cp -r /root/.cache/kagglehub/datasets/rishabhrp/chest-x-ray-dataset/versions/1 /content/chest-x-ray-dataset
dataset_path = "/content/chest-x-ray-dataset"
```
### Start building model pipeline
### Step 1: Data Preprocessing and Augmentation
In this step, we focus on the essential tasks required to prepare the Chest X-ray dataset for model training. This includes data loading, class balancing, and applying augmentations. Here's a breakdown of the process:

1.1. Loading the Dataset
Setting Dataset Paths:

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
Filtering Images Based on Availability:

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
1.2 Handling Multi-Label Format:

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
1.3 One-Hot Encoding the Labels:

The multi-label format is then converted into a binary matrix using MultiLabelBinarizer. This means each label (like "Atelectasis", "Effusion", etc.) becomes its own column, and a 1 or 0 is assigned depending on whether the label is present or not for each image.
```
mlb = MultiLabelBinarizer()
df_labels = pd.DataFrame(mlb.fit_transform(df["Finding Labels"]), columns=mlb.classes_)
```
The resulting binary matrix (df_labels) is concatenated with the original DataFrame (df), so now each row has the image index and the corresponding binary labels.
```
df = pd.concat([df, df_labels], axis=1)
```
1.4 Dataset Splitting:

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

1.5 Image Augmentation
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
