import os
import logging
from dotenv import load_dotenv

def load_env_and_init_logging():
    """Load environment variables and configure logging."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )