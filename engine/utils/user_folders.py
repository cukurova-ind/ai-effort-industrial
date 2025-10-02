import os
from django.conf import settings


def get_user_folder_path(user, subfolder=None):
    """
    Get the user-specific folder path
    
    Args:
        user: Django User object
        subfolder: Optional subfolder name (e.g., 'profiles', 'checkpoints')
        
    Returns:
        Full path to user's folder or subfolder
    """
    safe_folder_name = user.email.replace("@", "_at_").replace(".", "_dot_")
    base_path = os.path.join(settings.MEDIA_ROOT, "modeling", safe_folder_name)
    
    if subfolder:
        return os.path.join(base_path, subfolder)
    return base_path


def ensure_user_folders(user):
    """
    Ensure all required user folders exist
    
    Args:
        user: Django User object
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        base_path = get_user_folder_path(user)
        
        # Create base folder
        os.makedirs(base_path, exist_ok=True)
        
        # Create required subfolders
        subfolders = ["profiles", "checkpoints", "saved_models", "generated_images", "upload_images"]
        for subfolder in subfolders:
            folder_path = os.path.join(base_path, subfolder)
            os.makedirs(folder_path, exist_ok=True)
            
        return True
    except Exception as e:
        print(f"Error creating user folders for {user.username}: {e}")
        return False


def user_folder_exists(user, subfolder=None):
    """
    Check if user folder exists
    
    Args:
        user: Django User object
        subfolder: Optional subfolder name
        
    Returns:
        bool: True if folder exists
    """
    folder_path = get_user_folder_path(user, subfolder)
    return os.path.exists(folder_path)