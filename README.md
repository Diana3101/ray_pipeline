# How to run and test ray serve over the yolo_v7 model

## Running a Ray Serve Application
- Use requirements.txt to install dependencies
- Go to yolo_v7 folder
- Upload [**YOLOv7 weights**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) to the yolo_v7 folder
- run the following command to start ray serve app:
```bash
serve run object_detector_serve:detector
```

## Test model over HTTP
### Test one-image request and batch request
- While detector is running, open a separate terminal window 
- Go to yolo_v7 folder
- Run the client script:
```bash
python object_detector_client.py
```
### Test request to batch_handler
- While detector is running, open a separate terminal window 
- Go to yolo_v7 folder
- Run the client script:
```bash
python object_detector_client_batch.py
```
### Load testing using Locust (one-image request OR batch request)
- While detector is running, open a separate terminal window 
- Go to yolo_v7 folder
- Run the following command:
```bash
locust -f locust_test.py
```
