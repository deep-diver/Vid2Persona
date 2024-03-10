import os

def get_env_vars():
    gcp_project_id = os.getenv("GCP_PROJECT_ID", None)
    gcp_project_loc = os.getenv("GCP_PROJECT_LOCATION", None)

    return gcp_project_id, gcp_project_loc