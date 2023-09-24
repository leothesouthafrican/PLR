import os
import datetime

def create_output_folders():
    """ Create a sub-folder in `output` based on current date and time. """
    
    # Create the base output directory if it doesn't exist
    if not os.path.exists("output"):
        os.mkdir("output")

    # Create a sub-folder in `output` based on current date and time
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    sub_folder_path = os.path.join("output", current_time)
    os.mkdir(sub_folder_path)
    
    return sub_folder_path