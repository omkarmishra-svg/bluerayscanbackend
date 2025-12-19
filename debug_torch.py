try:
    print("Importing torch...")
    import torch
    print(f"Torch version: {torch.__version__}")
    print("Torch import successful")
except Exception as e:
    print(f"Torch import failed: {e}")

try:
    print("\nImporting transformers...")
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except Exception as e:
    print(f"Transformers import failed: {e}")
