#!/usr/bin/env python3
"""
Main entry point for Vercel deployment
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the main dashboard
from merged_dashboard import main

if __name__ == "__main__":
    main()