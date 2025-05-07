# AIHackathon-Team-Buraq

## Folder Structure
- **backend**: Contains the backend code written in FastAPI.
  - **images**: Folder to store input and output images. Create this folder before running the backend.
  - **models**: Contains pre-trained models required for the backend pipeline:
    - `YOLO_element_best.pt`: YOLO model for component detection.
    - `YOLO_text_best.pt`: YOLO model for text region detection.
    - `crnn_inference_model.h5`: CRNN model for OCR.

- **frontend**: Contains the Flutter mobile app code.

## Backend Setup (FastAPI)
1. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. Install the required system package:
   ```bash
   sudo apt update
   sudo apt install libngspice0-dev
   ```
3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install the required Python libraries:
   ```bash
   pip install fastapi uvicorn opencv-python-headless google-generativeai numpy tensorflow ultralytics scikit-learn scikit-image pillow pyspice
   ```
5. Create an `images` folder:
   ```bash
   mkdir images
   ```
6. Ensure the `models` folder contains the required pre-trained models:
   - `YOLO_element_best.pt`
   - `YOLO_text_best.pt`
   - `crnn_inference_model.h5`

7. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
8. The server will be available at `http://127.0.0.1:8000`.

## Frontend Setup (Flutter)
1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Ensure you have Flutter installed. If not, follow the [Flutter installation guide](https://flutter.dev/docs/get-started/install).
3. Install the required dependencies:
   ```bash
   flutter pub get
   ```
4. Run the Flutter app on a connected device or emulator:
   ```bash
   flutter run
   ```
5. The app will launch on your device or emulator.
