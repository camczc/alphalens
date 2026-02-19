"""
scripts/init_db.py

Run once to create all tables:
    python scripts/init_db.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import init_db

if __name__ == "__main__":
    print("Creating database tables...")
    init_db()
    print("Done! Tables created successfully.")
