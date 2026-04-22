import os

print("OPENAI:", "FOUND" if os.environ.get("OPENAI_API_KEY") else "MISSING")
print("EBIRD:", "FOUND" if os.environ.get("EBIRD_API_KEY") else "MISSING")