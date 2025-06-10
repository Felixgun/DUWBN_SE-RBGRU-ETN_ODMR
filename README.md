# T-SE-Res-Bi-GRU: Gesture Control for Mobile Robots  
Transforming UWB signals into robot actions through deep sequential learning.

![Model Overview](./images/Tr-SE-Res-Bi-GRU_Diagram.png)  
*Figure 1 — The T-SE-Res-Bi-GRU architecture*

---

## What This Is

This project enables **gesture-based control** of mobile robots using signal data from a **Distributed Ultra-Wideband (DUWB)** network.

It leverages a deep learning model — **T-SE-Res-Bi-GRU** — trained to interpret UWB signal sequences as specific gesture commands. The system translates these into real-time movement instructions for a mobile robot.


---

## Core Workflow
![Gsture Shape](./images/gestures.png)  
*Figure 2 — The T-SE-Res-Bi-GRU architecture*
**From human gesture to mobile robot action:**

1. A user wears a UWB tag on their hand.
2. Fixed UWB anchors record signal responses as gestures are performed.
3. The signal sequence is fed to the model
4. The output is mapped to a robot command (e.g., forward, stop, turn).

---

## System Components

![device](./images/photo.png)  
*Figure 1 — The T-SE-Res-Bi-GRU architecture*

- **Wearable UWB Tag** — worn on the hand  
- **Multiple UWB Anchors** — fixed around the environment  
- **Jetson NX or PC** — runs inference in real time  
- **Mobile Robot** — receives commands based on predicted gestures  
- **Pretrained T-SE-Res-Bi-GRU model** — the gesture recognizer

---

## Project Structure

| File               | Purpose                                      |
|--------------------|----------------------------------------------|
| `train and val.py` | Train the model using collected gesture data |
| `inference-pc.py`  | Run inference on standard computers          |
| `inference-jetson.py` | Optimized inference for Jetson devices    |
| `dataset.zip`      | Training and testing dataset (UWB signal sequences)      |

---


## Dataset Info

Collected using DUWB with labeled gestures, the dataset includes:
- Time-series signals from 4+ UWB anchors
- Labels for each gesture performed (e.g. "left", "right", "stop")

**Want to build your own dataset?**  
A collection script (`inference_pc.py`) is included to help you record new samples.

---


## Example Output

![Robot Output](./result.png)  
*Figure 2 — Mobile robot trajectory controlled via gesture recognition*

The model enables smooth, real-time control with minimal latency. It performs well even when gestures are performed at varying speeds or orientations.

---



## Maintainer

**Felix Gunawan**  
📫 [felix.iniemail@yahoo.com](mailto:felix.iniemail@yahoo.com)  
🔗 [linkedin.com/in/felixg26](https://linkedin.com/in/felixg26)  
🐙 [github.com/Felixgun](https://github.com/Felixgun)


