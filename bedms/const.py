"""
This module contains constant values used in the 'bedms' package.
"""

PROJECT_NAME = "bedmess"

AVAILABLE_SCHEMAS = ["ENCODE", "FAIRTRACKS", "BEDBASE"]
PEP_FILE_TYPES = ["yaml", "csv"]
REPO_ID = "databio/attribute-standardizer-model6"
MODEL_ENCODE = "model_encode.pth"
MODEL_FAIRTRACKS = "model_fairtracks.pth"
MODEL_BEDBASE = "model_bedbase.pth"
ENCODE_VECTORIZER_FILENAME = "vectorizer_encode.pkl"
FAIRTRACKS_VECTORIZER_FILENAME = "vectorizer_fairtracks.pkl"
BEDBASE_VECTORIZER_FILENAME = "vectorizer_bedbase.pkl"
ENCODE_LABEL_ENCODER_FILENAME = "label_encoder_encode.pkl"
FAIRTRACKS_LABEL_ENCODER_FILENAME = "label_encoder_fairtracks.pkl"
BEDBASE_LABEL_ENCODER_FILENAME = "label_encoder_bedbase.pkl"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
HIDDEN_SIZE = 32
DROPOUT_PROB = 0.113
CONFIDENCE_THRESHOLD = 0.70
EMBEDDING_SIZE = 384
INPUT_SIZE_BOW_ENCODE = 10459
INPUT_SIZE_BOW_FAIRTRACKS = 13617
OUTPUT_SIZE_FAIRTRACKS = 15
OUTPUT_SIZE_ENCODE = 18
NUM_CLUSTERS = 3
INPUT_SIZE_BOW_BEDBASE = 13708
OUTPUT_SIZE_BEDBASE = 12
