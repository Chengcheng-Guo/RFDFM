# -*- coding: UTF-8 -*-
# RFDFM source code
# May-22-2024
import glob
import os
from typing import List, Optional

import nibabel
import nrrd
import numpy as np
import h5py
import pandas as pd
import torch

from about_log import logger




def loadh5(path):
  with h5py.File(path, 'r') as h5f:
    dataset = h5f['my_dataset']
    return np.array(dataset)

class custom_data(torch.utils.data.dataset.Dataset):
  def __init__(self, file_dir, data_list, aug_prob=None):
    super(custom_data, self).__init__()
    self.file_dir = file_dir
    self.files = os.listdir(file_dir)

    self.data_list = data_list
    with open(data_list, "r") as f:
      index_ = f.readlines()
    index = []
    for i in index_:
      i = i.strip()
      if i.startswith('sample') and (i.endswith(".npy") or i.endswith(".h5")):
        index.append(i)

    self.index = list()
    csv_file = "./Lung_.csv"
    csv = pd.read_csv(csv_file, encoding="gbk")
    csv = csv[~pd.isna(csv.LUAD_subtype)]
    csv = csv.sort_values(by="ID")
    csv = csv.reset_index(drop=True)
    for file in index:
      ID = file.split("sample")[-1].split("_")[0]
      spacing = float(file.split("spacing")[-1].replace('.npy', '').replace('.h5', ''))
      tmp = csv[csv.ID == ID]
      for row in tmp.itertuples():
        rad = 10.0 * row._9 / spacing / 2.0
        xmin = round(max(float(row.X) - rad, 0.0))
        xmax = round(min(float(row.X) + rad, 511.0))
        ymin = round(max(float(row.Y) - rad, 0.0))
        ymax = round(min(float(row.Y) + rad, 511.0))
        zmin = round(float(row.Z.split('-')[0]))
        zmax = round(float(row.Z.split('-')[1]))
        self.index.append({"rad": rad, "xmin": xmin, "xmax": xmax,
                           "ymin": ymin, "ymax": ymax,
                           "zmin": zmin, "zmax": zmax,
                           "x": round(row.X), "y": round(row.Y),
                           "file": file_dir + "/" + file, "mask": file_dir.replace("/ct", "/mask") + "/" + file.replace(".npy", ".h5"),
                           "cls": int(row.LUAD_subtype == "IAC")})
    
    # data augment (if any)
    # 0. original
    # 1. rotate 90
    # 2. rotate 180
    # 3. rotate 270
    # 4. vertical flip
    # 5. horizontal flip
    # 6. flip by z-axis

    self.aug_func = {
        0: lambda x: x,
        1: lambda x: np.transpose(x, axes=(0, 1, 3, 2))[:, :, :, ::-1].copy(),
        2: lambda x: x[:, :, ::-1, ::-1].copy(), 
        3: lambda x: np.transpose(x, axes=(0, 1, 3, 2))[:, :, ::-1].copy(),
        4: lambda x: x[:, :, ::-1].copy(),
        5: lambda x: x[:, :, :, ::-1].copy(),
        6: lambda x: x[:, ::-1].copy(),
    }
     
    if aug_prob is None or len(aug_prob) == len(self.aug_func):
       self.aug_prob = aug_prob
    else:
       self.aug_prob = None
       logger.error(f"Unacceptable data augmentation parameters {aug_prob}! Run w/o augumentation!")

  def __getitem__(self, i):
    try:
      if self.index[i]["file"].endswith(".h5"):
        f = (loadh5(self.index[i]["file"]) + 400).astype(np.float32) / 1800.0
      else:
        f = (np.load(self.index[i]["file"]) + 400).astype(np.float32) / 1800.0
    except:
      print(f"Failed to load {self.index[i]['file']}")

    try:
        if self.index[i]["mask"].endswith(".h5"):
          m = loadh5(self.index[i]["mask"])
        else:
          m = np.load(self.index[i]["mask"])
    except:
      print(f"Failed to load {self.index[i]['mask']}")
    

    
    if self.index[i]["xmax"] - self.index[i]["xmin"] >= 32:
      xstart = max(0, min(f.shape[1] - 32, np.random.randint(self.index[i]["xmin"] - 2, self.index[i]["xmax"] + 2 - 32, size=1).item()))
    else:
      xstart = max(0, min(f.shape[1] - 32, np.random.randint(self.index[i]["xmax"] - 32 - 2, self.index[i]["xmin"] + 2, size=1).item()))
    if self.index[i]["ymax"] - self.index[i]["ymin"] >= 32:
      ystart = max(0, min(f.shape[2] - 32, np.random.randint(self.index[i]["ymin"] - 2, self.index[i]["ymax"] + 2 - 32, size=1).item()))
    else:
      ystart = max(0, min(f.shape[2] - 32, np.random.randint(self.index[i]["ymax"] - 32 - 2, self.index[i]["ymin"] + 2, size=1).item()))
    if self.index[i]["zmax"] - self.index[i]["zmin"] >= 32:
      zstart = max(0, min(f.shape[0] - 32, np.random.randint(self.index[i]["zmin"] - 2, self.index[i]["zmax"] + 2 - 32, size=1).item()))
    else:
      zstart = max(0, min(f.shape[0] - 32, np.random.randint(self.index[i]["zmax"] - 32 - 2, self.index[i]["zmin"] + 2, size=1).item()))
    #print(xstart, ystart, zstart)
      
    f = f[np.newaxis, zstart:(zstart+32), ystart:(ystart+32), xstart:(xstart+32)].astype(np.float32)
    m = m[np.newaxis, zstart:(zstart+32), ystart:(ystart+32), xstart:(xstart+32)].astype(np.int32)
    
    if self.aug_prob is not None:
      aug_id = np.random.choice(len(self.aug_func), 1, p=self.aug_prob).item()
      # print(aug_id)
      return {"image": self.aug_func[aug_id](f),
              "mask": self.aug_func[aug_id](m),
              "label": self.index[i]["cls"]}
    else:
      return {"image": f,
              "mask": m,
              "label": self.index[i]["cls"]}

  def __len__(self):
    return len(self.index)
  

class custom_data_val(custom_data):
  def __getitem__(self, i):
    try:
      if self.index[i]["file"].endswith(".h5"):
        f = (loadh5(self.index[i]["file"]) + 400).astype(np.float32) / 1800.0
      else:
        f = (np.load(self.index[i]["file"]) + 400).astype(np.float32) / 1800.0
    except:
      print(f"Failed to load {self.index[i]['file']}")

    try:
        if self.index[i]["mask"].endswith(".h5"):
          m = loadh5(self.index[i]["mask"])
        else:
          m = np.load(self.index[i]["mask"])
    except:
      print(f"Failed to load {self.index[i]['mask']}")
    

    xstart = max(0, min(f.shape[1] - 32, self.index[i]["x"] - 16))
    ystart = max(0, min(f.shape[2] - 32, self.index[i]["y"] - 16))
    zstart = max(0, min(f.shape[0] - 32, round((self.index[i]["zmin"] + self.index[i]["zmax"]) / 2.0) - 16))

    return {"image": f[np.newaxis, zstart:(zstart+32), ystart:(ystart+32), xstart:(xstart+32)].astype(np.float32),
            "mask": m[np.newaxis, zstart:(zstart+32), ystart:(ystart+32), xstart:(xstart+32)].astype(np.int32),
            "label": self.index[i]["cls"]}


class auxiliary_data(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir, data_list):
        super(auxiliary_data, self).__init__()
        self.file_dir = file_dir
        self.files = os.listdir(file_dir)

        self.data_list = data_list
        with open(data_list, "r") as f:
            index_ = f.readlines()
        index = []
        for i in index_:
            i = i.strip()
            if i.startswith('sample') and (i.endswith(".npy") or i.endswith(".h5")):
                index.append(i)

        self.index = list()
        csv_file = "/course75/RealData/Lung_SegAndCls_20230907.csv"
        csv = pd.read_csv(csv_file, encoding="gbk")
        csv = csv[~pd.isna(csv.LUAD_subtype)]
        csv = csv.sort_values(by="ID")
        csv = csv.reset_index(drop=True)
        for file in index:
            ID = file.split("sample")[-1].split("_")[0]
            spacing = float(file.split("spacing")[-1].replace('.npy', '').replace('.h5', ''))
            tmp = csv[csv.ID == ID]
            for row in tmp.itertuples():
                rad = 10.0 * row._9 / spacing / 2.0
                xmin = int(max(float(row.X) - rad, 0.0))
                xmax = int(min(float(row.X) + rad, 511.0))
                ymin = int(max(float(row.Y) - rad, 0.0))
                ymax = int(min(float(row.Y) + rad, 511.0))
                zmin = int(row.Z.split('-')[0])
                zmax = int(row.Z.split('-')[1])
                self.index.append({"rad": rad, "xmin": xmin, "xmax": xmax,
                                   "ymin": ymin, "ymax": ymax,
                                   "zmin": zmin, "zmax": zmax,
                                   "file": file_dir + "/" + file,
                                   "mask": file_dir.replace("/ct", "/mask") + "/" + file.replace(".npy", ".h5"),
                                   "cls": int(row.LUAD_subtype == "IAC")})

    def __getitem__(self, i):
        try:
            if self.index[i]["file"].endswith(".h5"):
                f = (loadh5(self.index[i]["file"]) + 400).astype(np.float32) / 1800.0
            else:
                f = (np.load(self.index[i]["file"]) + 400).astype(np.float32) / 1800.0
        except:
            print(f"Failed to load {self.index[i]['file']}")

        try:
            if self.index[i]["mask"].endswith(".h5"):
                m = loadh5(self.index[i]["mask"])
            else:
                m = np.load(self.index[i]["mask"])
        except:
            print(f"Failed to load {self.index[i]['mask']}")

        if self.index[i]["xmax"] - self.index[i]["xmin"] >= 256:
            xstart = max(0, min(f.shape[1] - 256,
                                np.random.randint(self.index[i]["xmin"] - 2, self.index[i]["xmax"] + 2 - 256,
                                                  size=1).item()))
        else:
            xstart = max(0, min(f.shape[1] - 256,
                                np.random.randint(self.index[i]["xmax"] - 256 - 2, self.index[i]["xmin"] + 2,
                                                  size=1).item()))
        if self.index[i]["ymax"] - self.index[i]["ymin"] >= 256:
            ystart = max(0, min(f.shape[2] - 256,
                                np.random.randint(self.index[i]["ymin"] - 2, self.index[i]["ymax"] + 2 - 256,
                                                  size=1).item()))
        else:
            ystart = max(0, min(f.shape[2] - 256,
                                np.random.randint(self.index[i]["ymax"] - 256 - 2, self.index[i]["ymin"] + 2,
                                                  size=1).item()))
        if self.index[i]["zmax"] - self.index[i]["zmin"] >= 8:
            zstart = max(0, min(f.shape[0] - 8,
                                np.random.randint(self.index[i]["zmin"] - 2, self.index[i]["zmax"] + 2 - 8,
                                                  size=1).item()))
        else:
            zstart = max(0, min(f.shape[0] - 8,
                                np.random.randint(self.index[i]["zmax"] - 8 - 2, self.index[i]["zmin"] + 2,
                                                  size=1).item()))
        # print(xstart, ystart, zstart)
        return {"image": f[np.newaxis, zstart:(zstart + 8), ystart:(ystart + 256), xstart:(xstart + 256)].astype(
            np.float32),
                "mask": m[np.newaxis, zstart:(zstart + 8), ystart:(ystart + 256), xstart:(xstart + 256)].astype(
                    np.int32)}

    def __len__(self):
        return len(self.index)