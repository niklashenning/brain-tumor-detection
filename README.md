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
The dataset contains 7023 MRI scans of brains taken from different angles. Some of the brains
are healthy and others have tumors of different types (Glioma, Meningioma, or Pituitary adenoma).
Most of the images are 512x512 px in size but some of them have different / unique sizes
and aspect ratios.

<ins>Glioma:</ins><br>
A glioma is a type of tumor that originates from glial cells in the brain or spinal cord.
Gliomas typically appear as areas of abnormal signal intensity. They may present as well-defined
or irregular masses, often with surrounding edema (swelling) in the brain tissue.

<ins>Meningioma:</ins><br>
A meningioma is a type of tumor that arises from the meninges, the protective membranes
that cover the brain and spinal cord. Meningiomas typically appear as well-defined,
extra-axial masses, meaning they are located outside the brain tissue itself.
They often have a characteristic "dural tail", which is a thickening of the dura mater
(the outer layer of the meninges) adjacent to the tumor.

<ins>Pituitary adenoma:</ins><br>
A pituitary adenoma is a tumor that arises from the pituitary gland, which is 
located at the base of the brain. A pituitary adenoma typically appears as a well-defined
mass in the region of the pituitary gland. The size of the adenoma can vary, and larger
tumors may cause displacement of surrounding structures, such as the optic chiasm. 
In some cases, the adenoma may also cause enlargement of the sella turcica,
the bony cavity that houses the pituitary gland.


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