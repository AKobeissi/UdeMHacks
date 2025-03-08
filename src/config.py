# Database configuration
DB_NAME = 'parasite_diagnosis.db'

# ML Model settings
MODEL_PATH = 'parasite_model.pth'

# User roles
ROLES = ["patient", "doctor", "technician", "admin"]

# Updated Parasite classes based on your dataset
PARASITE_CLASSES = [
    "Babesia",
    "Leishmania", 
    "Leukocyte",
    "Plasmodium",
    "RBC",  # Red Blood Cells
    "Toxoplasma",
    "Trichomonad",
    "Trypanosome",
    "No parasite detected"  # Still keep this as an option
]

# Default accounts for testing
DEFAULT_ADMIN = {
    "username": "admin",
    "password": "admin123",
    "full_name": "System Administrator",
    "email": "admin@parasite.org"
}

DEFAULT_DOCTOR = {
    "username": "doctor1",
    "password": "doctor123",
    "full_name": "Dr. Jane Smith",
    "email": "doctor@parasite.org"
}

# Upload directory
UPLOAD_DIR = "uploads"