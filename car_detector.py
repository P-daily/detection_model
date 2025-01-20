import torch
import cv2
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)


# Check if CUDA is available and move the model to GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)  # Move model to GPU if available, otherwise CPU

pathlib.PosixPath = temp

# Ustawienia detekcji
CONFIDENCE_THRESHOLD = 0.6  # Minimalny próg pewności dla detekcji

# Function to check if the detected class is "car"
def is_car(class_name):
    return class_name.lower() == "car"

def detect_and_display(video_source):
    # Otwieranie źródła wideo (kamera lub plik)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Nie można otworzyć źródła wideo.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Nie można odczytać klatki wideo lub koniec pliku.")
            break

        # Wykonywanie detekcji na klatce
        results = model(frame)

        # Pobieranie detekcji
        detections = results.pandas().xyxy[0]  # Wyniki w formacie pandas

        for _, row in detections.iterrows():
            # Filtruj detekcje poniżej ustalonego progu pewności
            if row['confidence'] < CONFIDENCE_THRESHOLD:
                continue

            # Get class name from the trained model
            class_name = row['name']

            # Check if the detected object is a car
            if is_car(class_name):
                # Współrzędne prostokąta
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = f"{class_name} ({row['confidence']:.2f})"

                # Rysowanie prostokąta i etykiety na obrazie
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Wyświetlanie obrazu
        cv2.imshow("YOLOv5 Car Detection", frame)

        # Zakończ program po naciśnięciu klawisza 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Zwolnienie zasobów
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_source = r"films/kolor_1080p.mp4"
    detect_and_display(video_source)
