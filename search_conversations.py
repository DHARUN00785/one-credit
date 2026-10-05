import os
import re

directory = r"C:\Users\HP\.gemini\antigravity\conversations"
query_pattern = re.compile(r"def get_prediction_array|StandardScaler|MinMaxScaler|readmission_model", re.IGNORECASE)

for filename in os.listdir(directory):
    if filename.endswith(".pb"):
        path = os.path.join(directory, filename)
        try:
            with open(path, "rb") as f:
                content = f.read()
            # Try to decode as ascii/utf-8 with errors ignored to search text strings
            text = content.decode("utf-8", errors="ignore")
            matches = query_pattern.findall(text)
            if matches:
                print(f"\nFound matches in {filename}: {set(matches)}")
                # Find occurrences and print some surrounding context
                for m in re.finditer(query_pattern, text):
                    start = max(0, m.start() - 200)
                    end = min(len(text), m.end() + 500)
                    print(f"--- Context ---")
                    print(text[start:end])
                    print("=" * 40)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
