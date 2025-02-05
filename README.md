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
CSV File Loading: The dataset consists of a CSV file (Ground_Truth.csv) that contains image filenames along with associated labels for each X-ray image. This CSV file is read using Pandas.
```
df = pd.read_csv(csv_path)
```
Image Directory Setup: The images are located in the xray_images directory, and we load them using OpenCV (cv2).

```
images_path = os.path.join(dataset_path, "xray_images")
```
1.2. Handling Class Imbalance
The dataset may contain classes that occur too infrequently, potentially leading to class imbalance issues. To resolve this, we filter out labels that appear only once in the dataset. This is achieved by:
Counting occurrences of each label.
Removing labels that have only one occurrence.
```
multi_label_counts = df.iloc[:, 1].value_counts()
classes_to_remove = multi_label_counts[multi_label_counts == 1].index
df = df[~df.iloc[:, 1].isin(classes_to_remove)]
```





