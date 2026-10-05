import os

directory = r"C:\Users\HP\.gemini\antigravity\brain"
query = "get_prediction_array"

for root, dirs, files in os.walk(directory):
    if "ca3ff847-a6f2-433a-ad26-6e36b64f6ea2" in root:
        continue
    for filename in files:
        if not any(filename.endswith(ext) for ext in [".png", ".webp", ".zip", ".exe", ".keras", ".h5", ".pb", ".lnk", ".url"]):
            path = os.path.join(root, filename)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if query in content:
                    print(f"\nFound match in: {path}")
                    lines = content.splitlines()
                    for idx, line in enumerate(lines):
                        if query in line:
                            start_line = max(0, idx - 5)
                            end_line = min(len(lines), idx + 25)
                            print(f"--- Context (lines {start_line}-{end_line}) ---")
                            print("\n".join(lines[start_line:end_line]))
                            print("=" * 40)
            except Exception as e:
                pass
