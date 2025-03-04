# Setup
## Install libraries
```bash
pip install -r requirements.txt
```
<br/>

# Training pipeline (manual steps)
1. Record yourself in a mp4/mov video file (FullHD if possible @ 30fps)
    * In Windows you can use the Camera app
    * Make sure you include the gestures you want to train the model on.
        * The more examples of a gesture you have, the better the model will be able to recognize it.
        * Don't forget to include a "no gesture" example 
    * Place the file in a folder such as `movies/`
1. Use `parseVideo.py` to extract features from the video
    * Set the parameters:
        * `input_file_path` for the video file 
        * and `csv_output_file` for the output csv file
    * Run it in bash with
        ```bash
        python parseVideo.py
        ``` 
        This will create the output csv file, containing the video file path and the list of selected features
1. Use `analyzeMovement.py` to segment the video and save the gestures in a folder structure
    * Set the parameters:
        * `movement_file` should be the `csv_output_file` you've set earlier in `parseVideo.py`
        * `samples_folder` should be the folder where the segmented gestures will be saved
    * Run 
        ```bash
        python analyzeMovement.py
        ```
        This will segment the video and save the gestures in a folder structure
1. Use `trainModel.py` to train the model
    * Set the parameter `data_folders` to be the folder containing the segmented gestures to be used for training
        * 
    * Run 
        ```bash
        python trainmodel.py
        ```
        This will train the model and save it in the `model/` folder under a folder named gru_`timestamp` which contains:
            * `gru_model.pth` - the trained model
            * `info_model.txt` - the metadata of the model
            * `scaler.pkl` - the scaler used to normalize the data
1. Use `useGruModeLiveOnReducedFeatures.py` to test the gesture detection
    * Set the parameter `metadata_filename` to be the `info_model.txt` of the output model folder from the previous step
    * Run 
        ```bash
        python useGruModeLiveOnReducedFeatures.py
        ```
        This will start the camera and detect the gestures in real time, printing the detected gesture in the console
1. Evaluate results and repeat steps if necessary
    * If the model is not performing well, you can:
        * Record more examples of the gestures you want to train the model on
        * Use a different set of features
        * Use a different model architecture
        * etc.
        
