<h1 align="center">Brain Tumor Detection</h1>

<p align="center">
    Machine learning model that is able to detect and classify brain tumors in MRI scans.
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/6bf473d8-5bea-454c-8930-f38f01a317c7" width="450">
</p>


## About
This basic machine learning model is trained on the
[Brain Tumor (MRI Scans) dataset](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans)
and is able to recognize and accurately classify brain tumors in MRI scans.
<br>
It utilizes PyTorch's neural networks and Tensor libraries for training the model, pandas for loading
and handling the dataset, Pillow for loading and converting the images, and Matplotlib
for visualizing the training results.


## Dataset
...


## Results
The neural network was trained for 20 epochs with a learning rate of 0.001 using Adam
as the optimizer and CrossEntropyLoss as the criterion. Training was done on 80%
of the dataset while the remaining 20% were used to validate the results.

The model achieved an accuracy of 94.23% on the validation data.

```
Test Accuracy: 94.23% (1324/1405)
```

| Epoch | Loss     |
|-------|----------|
| 1     | 0.655658 |
| 5     | 0.002233 |
| 10    | 0.000839 |
| 15    | 0.004388 |
| 20    | 0.000002 |

<img src="https://github.com/user-attachments/assets/97728e8e-b752-4d70-8eec-3143c35db4b6" width="500">


## License
This software is licensed under the [MIT license](LICENSE).