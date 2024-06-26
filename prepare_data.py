import os
import pandas as pd
from pathlib import Path
import argparse
import json
import shutil
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

from cv2 import imread

DATA_DIR = "./datasets/"
OUTPUT_DIR = "./dataset/"
NUM_FOLDS = 2
SEED = 43

def read_folder_dataset(df : list, data_dir : str):
    # Create csv file include image path and its label
    for label, leaf_type in enumerate(os.listdir(data_dir)):
        if (leaf_type != ".DS_Store"):
            for image in os.listdir(os.path.join(data_dir, leaf_type)):
                if (image.endswith(".jpg") or image.endswith(".png")) or image.endswith(".bmp"):
                    imgpath = os.path.join(data_dir, leaf_type, image)
                    sample = {}
                    sample["imgpath"] = imgpath
                    sample["label"] = label
                    sample["medical_leaf_name"] = leaf_type

                    df.append(sample)

    return df

def prepare_data():
    df = []
    df = read_folder_dataset(df,DATA_DIR)
    df = pd.DataFrame(df)
    print(df)
    df.to_csv(os.path.join(OUTPUT_DIR, "full_dataset.csv"))

    # Label encoder medical_leaf_name feature
    LE = LabelEncoder()
    df["label"] = LE.fit_transform(df["medical_leaf_name"])

    # Name Mapping
    LE_name_mapping = {i: l for i, l in enumerate(LE.classes_)}

    f = open(os.path.join(OUTPUT_DIR,"name_mapping.json"), "w")
    json.dump(LE_name_mapping, f)
    f.close()

    # Split train/set
    
    X_train, X_test, Y_train, Y_test = train_test_split(df,df["label"], stratify=df["label"], test_size = 0.2)

    df_train = X_train.reset_index()
    df_test = X_test.reset_index()
    
    # Stratifiled K Folds (4 folds)
    skf = StratifiedKFold(n_splits = NUM_FOLDS, shuffle= True, random_state=SEED)
    df_train["fold"] = -1

    for i, (train_index, val_index) in enumerate(skf.split(df_train, df_train["label"])):
        df_train.loc[val_index, "fold"] = i

    print(df_train.groupby(['fold', 'label'])['imgpath'].size())

    df_train.to_csv(os.path.join(OUTPUT_DIR, "train_dataset.csv"))
    df_test.to_csv(os.path.join(OUTPUT_DIR, "test_dataset.csv"))
    
    print(len(df_train))

    # Create Train/Test Folds
    for fold in range(NUM_FOLDS):
        fold_path = f"{OUTPUT_DIR}fold_{fold}/"
        train_path = f"{fold_path}train/"
        val_path = f"{fold_path}val/"
        
        os.makedirs(fold_path, exist_ok= True)
        os.makedirs(train_path, exist_ok= True)
        os.makedirs(val_path, exist_ok= True)

        train_df = df_train[df_train["fold"] != fold].reset_index(drop = True)
        val_df = df_train[df_train["fold"] == fold].reset_index(drop = True)

        for index, row in train_df.iterrows():   
            src_dir = row["imgpath"]
            img_name = src_dir.split("/")[-1]
            medical_leaf_name = row["medical_leaf_name"]

            destination_dir = f"{train_path}{medical_leaf_name}"
            os.makedirs(destination_dir, exist_ok=True)
            
            shutil.copy(src_dir, destination_dir)
        
        for index, row in val_df.iterrows():   
            src_dir = row["imgpath"]
            img_name = src_dir.split("/")[-1]
            medical_leaf_name = row["medical_leaf_name"]

            destination_dir = f"{val_path}{medical_leaf_name}"
            os.makedirs(destination_dir, exist_ok=True)
            
            shutil.copy(src_dir, destination_dir)
if __name__ == "__main__":
    if OUTPUT_DIR:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    prepare_data()
 