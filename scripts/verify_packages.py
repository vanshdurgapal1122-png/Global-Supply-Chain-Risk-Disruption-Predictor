"""
Verification script for Step 1: Package Installations.
Uses importlib.metadata to verify and display installed package versions.
"""

import sys
import importlib.metadata

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PACKAGES = [
    "streamlit",
    "pandas",
    "numpy",
    "xgboost",
    "scikit-learn",
    "transformers",
    "torch",
    "yfinance",
    "plotly",
    "folium",
    "streamlit-folium",
    "joblib"
]

def main():
    print("=" * 65)
    print("   STEP 1: PACKAGE INSTALLATION & ENVIRONMENT VERIFICATION")
    print("=" * 65)
    print(f" Python Executable : {sys.executable}")
    print(f" Python Version    : {sys.version.split()[0]}")
    print("-" * 65)
    print(f" {'Package Name':<20} | {'Status':<12} | {'Installed Version'}")
    print("-" * 65)

    all_passed = True
    for pkg in PACKAGES:
        try:
            ver = importlib.metadata.version(pkg)
            status = "[OK] PASSED"
            print(f" {pkg:<20} | {status:<12} | {ver}")
        except Exception as e:
            all_passed = False
            status = "[!] FAILED"
            print(f" {pkg:<20} | {status:<12} | Error: {e}")

    print("=" * 65)
    if all_passed:
        print(" [SUCCESS] All 12 dependencies verified in the virtual environment!")
    else:
        print(" [WARNING] Some dependencies could not be verified.")
    print("=" * 65)

if __name__ == "__main__":
    main()
