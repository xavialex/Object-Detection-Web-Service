import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

class TotalPeople:
    """Static variables accesible for the main and HTTP threads.
    
    Args:
    i (int): Total number of people detected.
    img (nparray): Encoded image with the results of the prediction.
    
    """
    i = 0
    img = 0

# Serving a web interface
class ObjectDetectionHandler(BaseHTTPRequestHandler, TotalPeople):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
    def _set_image_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'image/jpeg')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_GET(self):
        """Looks in the target URL for the 'img' keyword.
        If so, loads into it the processed image.
        If not, loads the total number of people detected.
        
        """
        if 'img' in self.path:
            self._set_image_headers()
            content = TotalPeople.img
            self.wfile.write(content[1].tobytes())
        elif 'total' in self.path:
            self._set_headers()
            message = str(TotalPeople.i)
            self.wfile.write(bytes(message, "utf8"))
        else:
            # Loading the resulting web and serving it again
            self._set_headers()
            html_file = open("ui.html", 'r', encoding='utf-8')
            source_code = html_file.read()
            self.wfile.write(bytes(source_code, "utf8"))
      
    # Overriding log messages    
    def log_message(self, format, *args):
        return
        
class ObjectDetectionThread(threading.Thread):
    def __init__(self, name):
        super(ObjectDetectionThread, self).__init__()
        self.name = name
        self._stop_event = threading.Event() 
      
    def run(self):
        server_address = ('127.0.0.1', 8080)
        self.httpd = HTTPServer(server_address, ObjectDetectionHandler)
        self.httpd.serve_forever()

    def stop(self):
       self.httpd.shutdown()
       self.stopped = True
       
    def stopped(self):
       return self._stop_event.is_set()
   