import re, base64, os
from pathlib import Path

WORKING_DIR = 'working'
DATA_URL = ""  # Set at runtime if needed
OUTPUT = Path(WORKING_DIR) / 'input_view.jpg'

os.makedirs(WORKING_DIR, exist_ok=True)

m = re.search(r'base64,([A-Za-z0-9+/=\n\r]+)', DATA_URL)
if not m:
    raise SystemExit('No base64 payload found in DATA_URL')

b = base64.b64decode(m.group(1))
OUTPUT.write_bytes(b)
print(f'Saved {OUTPUT} ({len(b)} bytes)')
