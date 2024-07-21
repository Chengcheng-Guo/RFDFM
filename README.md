# <center> RFDFM: A Deep Factorization Machine Network Model for Invasive Lung Adenocarcinoma Screening in CT Images

### Dataset preparations

CT images for each patient should be preprocessed into a grey-scale 3D NumPy array with shape of $W\times512\times512$ and should be stored in a folder named `ct`. NumPy file should be named correctly, for example, ` sample2_spacing0.751953125.npy`, where the integer number is the patient ID and the float number should be spacing referring to the physical distance between adjacent pixels in the image. Subsequently, you need to create another folder named `mask` in the same directory as the `ct` folder and store the binary images of the mask of the lung nodules corresponding to all the npy files of the `ct` folder under the same filenames in `mask`. 

Patients in your dataset should be splitted into three distinct sets of train, validation and test. You should create three txt files that list file names of patients in the corresponding set, which should be assigned into argument `--train` or `--val`.

For labelling the nodes in each set of CT images, another csv file needs to be prepared and its path written to line 41 of utils.py. The ID columns of the csv file are the patient IDs just mentioned, and each row represents a node, where columns X, Y, and Z reflect the location of the node. The column `Size.cm.` is the diameter. Column LUAD_subtype taking the value Pre-IA or IAC is the classification label.


### Train

```bash
python run_model.py --train ./train.txt --valid ./val.txt -j 6 --batch_size 10 --epochs 120 --save_dir ./save_path
```
Note that `-save_dir` is specified, and the trained model will be saved here. A pth file will be saved for each round of model training.

### Evaluation
```bash
python eval_model.py --train ./test.txt --valid ./test.txt --batch_size 64 -j 8 --model_path ./dave_path/xxx.pth 
```

* The list of data samples to be used in the evaluation needs to be specified under the parameter `--valid`, the content of the parameter `--train` does not work.
* The path of the previously trained model needs to be written under the parameter `--model_path`.
