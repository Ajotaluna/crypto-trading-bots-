import os
import json

SOURCE_DIR = r"nascent_scanner/data_monthly"
OUTPUT_FILE = "whitelist.json"

def generate():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source dir not found: {SOURCE_DIR}")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith("_15m.csv")]
    symbols = sorted([f.replace("_15m.csv", "") for f in files])
    
    print(f"🔍 Found {len(symbols)} symbols.")
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(symbols, f, indent=4)
        
    print(f"✅ Generated {OUTPUT_FILE} with {len(symbols)} pairs.")

if __name__ == "__main__":
    generate()
