# Crowd Management System

This project implements a **Crowd Flow Monitoring System** using a **CNN + LSTM** model trained on the **PETS2009 dataset**. The system detects and analyzes crowd movement and integrates with **NX Meta** for live video processing.

## Features
- **Preprocessing**: Extracts bounding box data from PETS2009 annotations.
- **Model Training**: Uses a CNN + LSTM architecture to predict crowd flow.
- **ONNX Conversion**: Converts trained models to ONNX format for deployment.
- **NX Meta Integration**: Sends processed results to NX Meta for visualization.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Venkatesan11/crowd-management-system.git
   cd crowd-management-system
