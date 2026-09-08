# OratioMX

> Real-time sign-language recognition project combining computer vision, machine learning and application engineering.

## 🎯 Objective

OratioMX explores the recognition of sign-language gestures from camera input and their conversion into text and/or voice through a mobile application and supporting backend services.

The project demonstrates experience across **computer vision, ML pipelines, APIs, mobile development and real-time processing**.

## 🏗️ High-level architecture

```text
Camera
  │
  ▼
MediaPipe / Keypoint Extraction
  │
  ▼
Temporal Gesture Model
  │
  ▼
Inference API
  │
  ├──► Text
  └──► Voice
        │
        ▼
   React Native App
```

## 🧠 ML pipeline

The project uses pose/hand/body landmark information rather than relying exclusively on raw RGB frames.

Typical pipeline:

1. Capture camera samples
2. Extract landmarks/keypoints
3. Normalize temporal data
4. Build training sequences
5. Train the recognition model
6. Evaluate predictions
7. Expose inference to the application

## 🛠️ Technology areas

| Area | Technologies |
|---|---|
| Mobile | React Native · Expo |
| Computer vision | MediaPipe |
| ML | Python · TensorFlow / Keras · PyTorch experiments |
| Backend | FastAPI / Node.js depending on service |
| Data | PostgreSQL / Supabase depending on environment |
| Real-time | WebRTC / PeerJS experiments |

## 🔬 Engineering focus

- Temporal gesture recognition
- Landmark/keypoint preprocessing
- Model evaluation
- Real-time inference
- API integration
- Mobile UX for accessibility
- Reproducible data-processing scripts

## 📁 ML workflow

```text
capture_samples.py
       │
       ▼
normalize_samples.py
       │
       ▼
crear_keypoints.py
       │
       ▼
entrenar_modelo.py
       │
       ▼
evaluar_modelo.py
       │
       ▼
Inference / Application
```

## 🚀 Project status

OratioMX is an experimental/academic engineering project. Components and model architecture may evolve as recognition quality, latency and usability are improved.

## 🔐 Security & privacy

Camera data and model inputs should be treated as sensitive application data. Development environments should use synthetic/test samples where possible, keep credentials outside source control and avoid committing personal recordings.

## 📌 Portfolio context

OratioMX complements the security-focused projects in this profile by demonstrating **AI engineering, computer vision and real-time application development**.

For DevSecOps / Cloud Security work, see:

- [AURONTEK](https://github.com/ezelpc/AURONTEK)
- [BastionGuard](https://github.com/ezelpc/BastionGuard)
- [SOAR Log Analyzer](https://github.com/ezelpc/soar-log-analyzer)

## 👤 Author

**Ezequiel Nahun Pérez** — ESCOM–IPN · DevSecOps · Cloud Security · Cybersecurity
