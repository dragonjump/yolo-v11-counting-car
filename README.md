
 
# Video Car Counter  
This repository counts video cars. You can use it for video or live stream.
This approach uses a simple line counter. Whenever the car crosses the line, we count them out. 



## Demo  
See `demo.gif`
[demo](demo.gif)
![demo](demo.gif)


## Setup 
Search online how setup conda. 
Once you setup, create a conda environment with Python 3.11:
```
conda create --name ultralytics-env python=3.11 -y
conda activate ultralytics-env
conda activate ultralytics-env
conda install -c pytorch pytorch torchvision torchaudio

```

You may adjust the parameter in the code accordingly. Do read the ultralytics doc.
```
# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    show_in=False, #  count in
    show_out=True, # count out
    conf=0.6, # confidence level
    tracker="bytetrack.yaml",
    region=region_points,  # pass region points
    model="yolo11n-obb.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    # classes=[0, 2],  # count specific classes i.e. person and car with COCO pretrained model.
 
)
```

## Video credit 
https://www.youtube.com/watch?v=Mp6klx9oeZs&pp=ygUlY29weXJpZ2h0IGZyZWUgZHJvbmUgdmlldyBjYXIgdHJhZmZpYw%3D%3D 


## Detail Doc   
1. Object counting - https://docs.ultralytics.com/guides/object-counting/#objectcounter-arguments
2. Object counting - https://docs.ultralytics.com/tasks/obb/#visual-samples
3. Region counting - https://docs.ultralytics.com/guides/region-counting/