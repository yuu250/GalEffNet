# GalEffNet
I am a co-first author of the paper "Estimation of Stellar Mass and Star Formation Rate Based on Galaxy Images." This repository provides the main code used in the paper. You can download the pretrained weights of GalEffNet and the test dataset used in the paper from [URL](https://pan.baidu.com/s/1BTKr35C0EQ2ynD1QySXi9Q) (code: 119l). Below is an example of how to use the proposed GalEffNet for inference.

## Requirements
- `tensorflow==2.8.0`
- `tensorflow_hub==0.15.0`

## Load the Pre-trained Model and Run the Inference

1. **Load the Pre-trained Model**
    ```python
    custom_objects = {'KerasLayer': hub.KerasLayer}
    model_path = "galeffnet.h5"
    final_model = tf.keras.models.load_model(model_path, custom_objects)
    ```

2. **Load Scalers for Predictions**
    ```python
    with open('scaler_sfr.pkl', 'rb') as file:
        scaler_sfr = pickle.load(file)

    with open('scaler_lgm.pkl', 'rb') as file:
        scaler_lgm = pickle.load(file)
    ```

3. **Load and Preprocess the Input Data**
    ```python
    file3 = h5py.File('test_set.h5', 'r')
    img3 = file3['processed_images']

    original_length = 152
    target_length = 64
    left = (original_length - target_length) // 2
    top = (original_length - target_length) // 2
    right = left + target_length
    bottom = top + target_length

    cropped_images = []
    for image_array in img3:
        cropped_image = image_array[:, top:bottom, left:right]
        cropped_images.append(cropped_image)

    img_rgb3 = [to_rgb.dr2_rgb(i, ['g', 'r', 'z']) for i in cropped_images]
    img_rgb3 = np.array(img_rgb3)
    ```
4. **Generate Predictions**
    ```python
    y = final_model.predict(np.array(img_rgb3))
    ```

For detailed code, please refer to the `inference.ipynb` file.

## Citation

If you use our work, please cite our paper as follows:

@article{10.1093/mnras/stae1271,
    author = {Zhong, Jing and Deng, Zhijie and Li, Xiangru and Wang, Lili and Yang, Haifeng and Li, Hui and Zhao, Xirong},
    title = "{Estimation of stellar mass and star formation rate based on galaxy images}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    pages = {stae1271},
    year = {2024},
    month = {05},
    issn = {0035-8711},
    doi = {10.1093/mnras/stae1271},
    url = {https://doi.org/10.1093/mnras/stae1271},
    eprint = {https://academic.oup.com/mnras/advance-article-pdf/doi/10.1093/mnras/stae1271/57708731/stae1271.pdf},
}






