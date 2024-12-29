import os

def download_hub_file(**kwargs) -> tuple:
    """
    Download a model from HuggingFace Hub and return path and blob names\n
    :param repo_link: `str` The HF repository to download from
    :param file_name: `str` Name of the specific file to download
    :return: `tuple` the default download folder and folder contents
    """
    try:
        os.environ['HUGGINGFACE_HUB_CACHE'] = str(os.getcwd())
        from huggingface_hub import hf_hub_download
        hf_hub_download(**kwargs)
    except ImportError as error_log:
        ImportError(f"{error_log} huggingface_hub not installed.")
    else:
        repo_id = kwargs.get('repo_id')
        if not kwargs.get('local_dir'):
            download_folder ='models--' + repo_id.replace('/', '--')
            folder_path_named = os.path.join(download_folder, 'blobs')
        else:
            download_folder = kwargs.get('local_dir')
            folder_path_named = os.path.join(download_folder)
        folder_contents = os.listdir(folder_path_named)
        return download_folder, folder_contents

