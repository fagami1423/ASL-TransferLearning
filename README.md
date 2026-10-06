# 🤟 ASL Alphabet Recognition with Transfer Learning

Real-time American Sign Language (ASL) alphabet recognition from a webcam, with **text-to-speech** output, so hand signs are turned into spoken letters.

Built with **TensorFlow/Keras**, **InceptionV3** transfer learning and **OpenCV**.

![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

## ✨ Features
- **Transfer learning:** an ImageNet-pretrained InceptionV3 backbone (frozen), with a new classification head (global average pooling → Dense 1024 → softmax)
- **28 classes:** letters A–Z plus `Space` and `Nothing`
- **Live webcam demo:** draws a region of interest, classifies the hand sign on keypress and shows the predicted letter on screen
- **Speech output:** reads each prediction aloud with `pyttsx3`
- **Two model approaches:** the InceptionV3 transfer-learning model and a custom sequential CNN trained in Google Colab (`ASL_recognition.ipynb`)
- **Docker support** for a reproducible environment

## 🧠 How it works
```
Webcam frame ──► crop region of interest ──► resize ──► CNN classifier ──► softmax
                                                                  │
                                              predicted letter ◄──┘
                                                     │
                                    on-screen overlay + text-to-speech
```

## 📁 Project structure
| File | Purpose |
|---|---|
| `cnnmodel.py` | `Model` class: dataset loading (80/20 split), InceptionV3 model definition, save/load, accuracy plots |
| `train.py` | Trains the transfer-learning model and saves it to `model/` |
| `predict.py` | Real-time webcam inference with text-to-speech |
| `ASL_recognition.ipynb` | Custom CNN trained in Google Colab |
| `inception_v3.ipynb` | InceptionV3 experiments |
| `Dockerfile` | Containerized environment |

## 🚀 Getting started
```bash
git clone https://github.com/fagami1423/ASL-TransferLearning.git
cd ASL-TransferLearning
pip install -r requirements.txt
```

**Train:** place the ASL alphabet images in `dataset/<class_name>/` folders, then run:
```bash
python train.py
```

**Run the webcam demo:**
```bash
python predict.py
```
Hold your hand inside the blue box, press **`s`** to classify the sign, and press **`q`** to quit.

**Docker:**
```bash
docker build -t sign-classifier .
docker run -it sign-classifier bash
```

## 🛣️ Roadmap
- [ ] Use MediaPipe hand landmarks for faster, background-independent recognition
- [ ] Continuous prediction (no keypress) and word building
- [ ] Web demo on Hugging Face Spaces

## 🛠️ Tech stack
Python · TensorFlow / Keras · InceptionV3 · OpenCV · NumPy · Matplotlib · pyttsx3 · Docker

## 👤 Author
**Raj Kumar Phagami**: [GitHub](https://github.com/fagami1423)
