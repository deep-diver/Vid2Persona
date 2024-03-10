import argparse
from vid2persona import init
from vid2persona.pipeline import vlm

def workflow(args):
    traits = vlm.get_traits(
        args.gcp_project_id, 
        args.gcp_project_location, 
        args.target_movie_clip, 
        args.prompt_tpl_path
    )
    print(traits)

def sanity_check(args):
    gcp_project_id, gcp_project_location = init.get_env_vars()

    if args.gcp_project_id is None: args.gcp_project_id = gcp_project_id
    if args.gcp_project_location is None: args.gcp_project_location = gcp_project_location

    if args.gcp_project_id is None or args.gcp_project_location is None:
        raise ValueError("gcp-project-id or gcp-project-location is missing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcp-project-id', type=str)
    parser.add_argument('--gcp-project-location', type=str)
    parser.add_argument('--target-movie-clip', type=str, default="assets/sample1.mp4")
    parser.add_argument('--prompt-tpl-path', type=str, default="vid2persona/prompts")
    args = parser.parse_args()

    sanity_check(args)
    workflow(args)