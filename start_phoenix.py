import phoenix as px
import time
import signal
import sys

# Function to handle graceful shutdown
def signal_handler(sig, frame):
    print('\nStopping Phoenix server gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print("Starting Phoenix server...")
# Launch the server. It will block the process.
session = px.launch_app() 

print(f"\n=======================================================")
print(f"  ✅ Phoenix Server is Running! Access at: {session.url}  ")
print(f"  Collector listening on http://localhost:4318/v1/traces")
print(f"  DO NOT close this window. Stop with Ctrl+C.          ")
print(f"=======================================================\n")

# Keep the main thread alive to serve the application
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Phoenix server terminated by user.")
finally:
    # Ensure cleanup is attempted
    # px.launch_app() handles a lot internally, but a clean exit is best
    sys.exit(0)