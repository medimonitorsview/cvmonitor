# Computer vision Server for MediView


This server is part of MediView Medical Monitor View system, and performs two tasks:

1. server side image processing (e.g. qr detection and genration, image marking)
2. cleaning client side ocr data

## API

- `v1/detect_codes`
    Gets an image returns qr codes in the structure:
    ```json
    [{"data":"xxx","top":1,"left":1,"bottom":100,"right":100},]
    ```
- `v1/align_image`
    Gets and image and returns an image (jpeg)

- `v1/run_ocr`
    gets:
    ```json
    {
        "image": "base64 jpeg encoded image",
        "segments":[
        {
            "top":0,
            "left":0,
            "bottom":100,
            "right":100,
            "name":"monitor-blood",
        },]
    }
    ```
    returns:
    ```json
    [ {"segment_name":"string","value":"string"},]
    ```

- `v1/qr_display/<monitorId>`
    gets the monitor qr image (jpeg)


- `v1/measurements/<device>`
    get the measurment names for a device type, e.g:
    ```json
    ["HR","Temp"]
    ```


## Install:

```bash
pip insatll -e .
```

## Test

```bash
pytest
```

## Run

```bash
cvmonitor
```

## Build docker:

```bash
docker build .
```

## Develop:

- Ubuntu 18.04 (16.04 & 19.10 may work)



```bash
# Install Dependancies:
sudo apt-get update && sudo apt-get install -yy wget libzbar0 libjpeg-turbo8-dev libz-dev python3-pip python3-venv
# Create a virutal enviornment (once)
python3 -venv ~/envs/cvmonitors/
# Clone the repo:
git clone https://github.com/medimonitorsview/cvmonitor && cd cvmonitor
# Activate virtuale enviroment (every time)
source  ~/envs/cvmonitors/bin/activate
# Install in dev mode
pip install -e .
# Run tests
pytset
# maybe install matplotlib some packages for easier development
pip install matplotlib pdbpp 
```


## Simulator

Start the docker with 

```bash
cvmonitor/generator/generate.py --help
```

To see options.

Basically you need the send option which will generate devices and send images to the server, so just run


To generate a new set of devices and send them to the server in url you set, run:

```bash
docker run <image-name> --net host cvmonitor/generator/generate.py --send --url <my-server-url>
```


To delete all devices from server run:

```bash
docker run <image-name> --net host cvmonitor/generator/generate.py --delete-all --url <my-server-url>
```

