#!/usr/bin/env python3
import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation_pipeline import main

if __name__ == "__main__":
    main()
