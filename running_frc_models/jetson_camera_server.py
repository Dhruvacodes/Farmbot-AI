#!/usr/bin/env python3
"""USB camera MJPEG stream server for Jetson."""

import argparse
import cv2
import io
import threading
import time
from pathlib import Path

try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
except ImportError:
    from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler


class CameraStreamHandler(BaseHTTPRequestHandler):
    camera = None
    frame = None
    lock = threading.Lock()

    def do_GET(self):
        if self.path == "/usb.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--boundary")
            self.end_headers()

            while True:
                with self.lock:
                    if self.camera.frame is not None:
                        self.wfile.write(b"--boundary\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(self.camera.frame)}\r\n\r\n".encode())
                        self.wfile.write(self.camera.frame)
                        self.wfile.write(b"\r\n")
                time.sleep(0.03)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress log messages


class CameraCapture:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def start(self):
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self.lock:
                    self.frame = jpeg.tobytes()
            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="USB camera MJPEG stream server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8090, help="Port")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    args = parser.parse_args()

    camera = CameraCapture(args.camera)
    camera.start()
    CameraStreamHandler.camera = camera

    server = HTTPServer((args.host, args.port), CameraStreamHandler)
    print(f"Camera server running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        camera.stop()
        print("Stopped")


if __name__ == "__main__":
    main()
