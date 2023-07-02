# How to run and test ray serve over the yolo_v7 model on GPU

## Running a Ray Serve Application
- Use requirements.txt to install dependencies
- Go to yolo_v7 folder
- Upload [**YOLOv7 weights**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) to the yolo_v7 folder
- Start a Ray Cluster: 
```bash
ray start --head --port=6379
```
- Start a Ray Serve app:
```bash
serve run object_detector_serve:detector
```
## How restart a Ray Serve Application
- Stop a Ray Cluster and all running processes:
```bash
ray stop
```
- Start a Ray Cluster: 
```bash
ray start --head --port=6379
```
- Start a Ray Serve app:
```bash
serve run object_detector_serve:detector
```

## Test a Ray Serve Application over HTTP

### Test connection and output format:
- While detector is running, open a separate terminal window 
- Go to yolo_v7 folder
- Run the client script:
```bash
python object_detector_client.py
```
### Load Testing using Locust
- While detector is running, open a separate terminal window 
- Go to yolo_v7 folder
- Run the following command:
```bash
locust -f locust_test.py
```

## Results of Load Testing
| batch_size | throughput (rps) | latency 50th perc (ms per image) | latency 95th perc (ms per image) |
| :-- | :-: | :-: | :-: |
| 32 | 11 | 85 | 95 |

**You can find detailed results of the experiments in the [**Google sheet**](https://docs.google.com/spreadsheets/d/1MJICCQJi-ZQDgATnPMgaednYdPSER0w4kr_0yPJeuj0/edit?usp=sharing)**