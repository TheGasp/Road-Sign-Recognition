# Road Sign Recognition

This project aims to detect and classify road signs using artificial intelligence, specifically with the YOLOv5 model trained on the GTSDB (German Traffic Sign Detection Benchmark) dataset.

## Authors

- Tomas Szabo  
- Gaspard Langlais  
- Herald Nkounkou  

## Project Structure

```
.
├── Projet_AIF-GTSDB.ipynb      # Main notebook (data prep, training, testing)
├── yolov5/                     # YOLOv5 source code
├── GTSDB/                      # Dataset (images and annotations)
├── SiteWeb/                    # web site                    
├── image_test.jpg              # Example test image
└── README.md                   # This file
```

## Front End
<img width="1372" alt="Capture d’écran 2025-06-03 à 18 38 34" src="https://github.com/user-attachments/assets/7c801574-f7c3-4452-9987-8beb84927713" />



## Installation

1. **Clone YOLOv5**  
   Download [YOLOv5](https://github.com/ultralytics/yolov5) and place the `yolov5` folder at the project root.

2. **Install dependencies**  
   From the root directory:
   ```bash
   pip install -r yolov5/requirements.txt
   ```

3. **Prepare the dataset**  
   - Download the GTSDB dataset and place the `.ppm` images and `gt.txt` file in `GTSDB/`.
   - Run the `Projet_AIF-GTSDB.ipynb` notebook to convert images, generate labels, and create the `gtsdb.yaml` file.

## Training

You can train the model from the notebook or with the following command:
```bash
cd yolov5
python train.py --img 1024 --batch 16 --epochs 30 --data gtsdb.yaml --weights yolov5s.pt --name gtsdb_yolov5s
```

## Testing and Detection

To test the trained model:
```bash
cd yolov5
python detect.py --weights runs/train/gtsdb_yolov5s/weights/best.pt --img 640 --conf 0.15 --source ../image_test.jpg --name test_output
```

## Results

- Training curves and confusion matrix are saved in `runs/train/gtsdb_yolov5s*/`.
- Annotated images are saved in `runs/detect/test_output*/`.

![output](https://github.com/user-attachments/assets/89182e3b-4877-4152-a46f-11992d4322f5)

## References

- [YOLOv5 - Ultralytics](https://github.com/ultralytics/yolov5)
- [GTSDB Dataset](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)

## License

This project is for educational purposes.  
YOLOv5 is licensed under AGPLv3.
