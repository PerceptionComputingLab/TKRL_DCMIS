# ------------------------------------------------------------------------------
# Module where paths should be defined.
# ------------------------------------------------------------------------------
import os

# Path where intermediate and final results are stored
# abs_path = os.path.abspath('.')
abs_path = "/Share8/zhuzhanshi/tkrl"
storage_path = "storage"
storage_data_path = os.path.join(abs_path, storage_path, "data")
source_data_path = "/Share8/zhuzhanshi/download/CL_origin/"

# Original data paths. TODO: set necessary data paths.
# original_data_paths = {'example_dataset_name': 'storage/data'}

# Login for Telegram Bot
telegram_login = {"chat_id": "TODO", "token": "TODO"}
