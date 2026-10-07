import socket
import json
import time

HOST = '127.0.0.1'
PORT = 9090

def start_server():
    """
    Goal:
        Initialize and run a local TCP socket bridge server to ingest historical bar
        data from a MetaTrader 5 (MT5) Expert Advisor (EA) running under Wine or Windows.

    Execution Principle:
        1. Socket Initialization: Binds an IPv4 TCP socket (`AF_INET`, `SOCK_STREAM`) to `127.0.0.1:9090`
           with `SO_REUSEADDR` enabled to avoid `EADDRINUSE` during quick restarts.
        2. Connection Handshake: Listens and blocks until an inbound connection is accepted from the MT5 EA.
        3. Protocol Query: Pauses briefly (2 seconds) to allow the EA client socket buffer to stabilize,
           then dispatches a formatted historical bar request (`HISTORY|SYMBOL|TIMEFRAME|COUNT\\n`).
        4. Stream Buffering: Enters a continuous receive loop reading up to 4096-byte segments into an
           accumulator string until a newline delimiter (`\\n`) demarcates message completion.
        5. Serialization & Logging: Deserializes the completed JSON payload into Python objects,
           prints pretty-printed bar data, and gracefully recovers from JSON parse or reset errors.
    """
    # Create a TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        # Allow port reuse so you don't get "Address already in use" errors if you restart quickly
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        
        print(f"Python Server listening on {HOST}:{PORT}...")
        
        # Wait for MT5 to connect
        conn, addr = server.accept()
        with conn:
            print(f"✅ MT5 EA connected from Wine! ({addr[0]}:{addr[1]})")
            
            # Wait 2 seconds before requesting, similar to the Node timeout
            time.sleep(2)
            
            # Format: COMMAND|SYMBOL|TIMEFRAME|COUNT\n
            request_cmd = "HISTORY|EURUSD|H1|5\n"
            print(f"Requesting historical data: {request_cmd.strip()}")
            conn.sendall(request_cmd.encode('utf-8'))
            
            data_buffer = ""
            
            # Continuously listen for incoming data
            while True:
                try:
                    # Receive up to 4096 bytes at a time
                    chunk = conn.recv(4096).decode('utf-8')
                    
                    if not chunk:
                        print("❌ MT5 EA disconnected")
                        break
                        
                    data_buffer += chunk
                    
                    # Check if MT5 sent the newline character signaling the end of the JSON
                    if '\n' in data_buffer:
                        complete_message = data_buffer.strip()
                        data_buffer = "" # Clear buffer for any future requests
                        
                        try:
                            # Parse the JSON string into a Python list of dictionaries
                            historical_data = json.loads(complete_message)
                            print("\n📈 Historical Data Received:")
                            
                            # Pretty print the JSON output
                            print(json.dumps(historical_data, indent=4))
                            
                            # If you want to ask for more data, you could send another command here.
                            
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON: {e}")
                            print(f"Raw data received: {complete_message}")
                            
                except ConnectionResetError:
                    print("❌ Connection was reset by MT5")
                    break
                except Exception as e:
                    print(f"Socket error: {e}")
                    break

if __name__ == "__main__":
    start_server()