# How to run and test ray serve over the yolo_v7 model

## Running a Ray Serve Application
- Use requirements.txt **in the parent folder** to install dependencies
- run the following command to start ray serve app:
```bash
serve run object_detector_serve:detector
```

## Test model over HTTP
- While detector is running, open a separate terminal window and run the client script:
```bash
python object_detector_client.py
```
