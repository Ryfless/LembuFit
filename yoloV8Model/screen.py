import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

def detect_image():
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    results = model.predict(source=file_path, conf=0.25, save=False)
    for r in results:
        im_bgr = r.plot()
        cv2.imshow("Hasil Deteksi Sapi (Foto)", im_bgr)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_video():
    file_path = filedialog.askopenfilename(
        title="Pilih Video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )
    if not file_path:
        return
    cap = cv2.VideoCapture(file_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.25, save=False)
        for r in results:
            frame = r.plot()
        cv2.imshow("Hasil Deteksi Sapi (Video)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def detect_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.25, save=False)
        for r in results:
            frame = r.plot()
        cv2.imshow("Hasil Deteksi Sapi (Webcam)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# GUI Setup
root = tk.Tk()
root.title("Deteksi Sapi - YOLOv8")
root.geometry("300x200")

tk.Label(root, text="Pilih Mode Deteksi", font=("Arial", 14)).pack(pady=10)
tk.Button(root, text="Deteksi dari Foto", command=detect_image, width=20).pack(pady=5)
tk.Button(root, text="Deteksi dari Video", command=detect_video, width=20).pack(pady=5)
tk.Button(root, text="Deteksi dari Webcam", command=detect_webcam, width=20).pack(pady=5)
tk.Button(root, text="Keluar", command=root.quit, width=20).pack(pady=10)

root.mainloop()
