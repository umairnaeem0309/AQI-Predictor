#!/usr/bin/env python3
"""Verify Python 3.11 environment has all required dependencies."""
import sys

def main():
    print(f"Python: {sys.version}")
    checks = [
        ("duckdb", "duckdb"),
        ("hopsworks", "hopsworks"),
        ("mlflow", "mlflow"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("pydantic", "pydantic"),
        ("requests", "requests"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
        ("responses", "responses"),
        ("pytest", "pytest"),
        ("xgboost", "xgboost"),
        ("evidently", "evidently"),
        ("yaml", "pyyaml"),
        ("dotenv", "python-dotenv"),
    ]
    all_ok = True
    for module_name, display_name in checks:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  {display_name}: {version}")
        except ImportError as e:
            print(f"  {display_name}: MISSING ({e})")
            all_ok = False

    if all_ok:
        print("\nALL DEPENDENCIES VERIFIED")
    else:
        print("\nSOME DEPENDENCIES MISSING")
        sys.exit(1)

if __name__ == "__main__":
    main()
