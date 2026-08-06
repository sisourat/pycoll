import sys
import io

class Tee:
    """Redirect stdout to both a file and the terminal."""
    def __init__(self, file_path):
        self.file = open(file_path, "w")
        self.terminal = sys.stdout  # Save the original stdout

    def write(self, message):
        self.terminal.write(message)  # Write to terminal
        self.file.write(message)     # Write to file
        self.flush()                  # Ensure output is flushed

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()
