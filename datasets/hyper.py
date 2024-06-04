import os
import tempfile
import pandas as pd
import glob
import torchvision
from tqdm.auto import tqdm

HYPER_CLASSES = {
    'cecum':0, 'ileum':1, 'retroflex-rectum':2, 'hemorrhoids':3, 'polyps':4,
       'ulcerative-colitis-grade-0-1':5, 'ulcerative-colitis-grade-1':6,
       'ulcerative-colitis-grade-1-2':7, 'ulcerative-colitis-grade-2':8,
       'ulcerative-colitis-grade-2-3':9, 'ulcerative-colitis-grade-3':10,
       'bbps-0-1':11, 'bbps-2-3':12, 'impacted-stool':13, 'dyed-lifted-polyps':14,
       'dyed-resection-margins':15, 'pylorus':16, 'retroflex-stomach':17, 'z-line':18,
       'barretts':19, 'barretts-short-segment':20, 'esophagitis-a':21,
       'esophagitis-b-d':22
    }

CLASSES = (
       'cecum', 'ileum', 'retroflex-rectum', 'hemorrhoids', 'polyps',
       'ulcerative-colitis-grade-0-1', 'ulcerative-colitis-grade-1',
       'ulcerative-colitis-grade-1-2', 'ulcerative-colitis-grade-2',
       'ulcerative-colitis-grade-2-3', 'ulcerative-colitis-grade-3',
       'bbps-0-1', 'bbps-2-3', 'impacted-stool', 'dyed-lifted-polyps',
       'dyed-resection-margins', 'pylorus', 'retroflex-stomach', 'z-line',
       'barretts', 'barretts-short-segment', 'esophagitis-a',
       'esophagitis-b-d'
)
root = "/storage/vatsal/datasets/hyper"
root_new = "/storage/vatsal/datasets/hyper_diffusion"

def main():

    for split in ["train", "test"]:
        out_dir = os.path.join(root_new,split)
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("processing...")
        image_paths = glob.glob(f'{root}/labeled-images/*/*/*/*.jpg')
        labels = pd.read_csv(f'{root}/labeled-images/image-labels.csv')
        split = pd.read_csv('https://raw.githubusercontent.com/simula/hyper-kvasir/master/official_splits/2_fold_split.csv',sep=';')
        idx = 0 if split=="train" else 1
        paths = list(split[split['split-index'] == idx]['file-name'])
        image_paths = [path for path in image_paths if path.split('/')[-1] in paths]
        targets = []
        for image_path in image_paths:
            targets.append(HYPER_CLASSES[labels[labels['Video file'] == image_path.split('/')[-1].split('.')[0]]['Finding'].reset_index(drop=True)[0]])


        print("dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(image_paths))):
            image, label = image_paths[i], targets[i]
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)


if __name__ == "__main__":
    main()
