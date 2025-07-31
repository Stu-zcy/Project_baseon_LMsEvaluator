from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return "SocketIO is running."

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=50001, debug=False)