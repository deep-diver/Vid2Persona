import argparse
from vid2persona import init
from vid2persona.pipeline import vlm

def validate_args(func):
    def inner_function(args):
        if args.gcp_project_id is None or args.gcp_project_location is None:
            raise ValueError("gcp-project-id or gcp-project-location is missing")
        return func(args)
    return inner_function

@validate_args
def workflow(args):
    # Assuming vlm.get_traits is a function you've defined elsewhere
    traits = vlm.get_traits(
        args.gcp_project_id, 
        args.gcp_project_location, 
        args.target_movie_clip, 
        args.prompt_tpl_path
    )
    print(traits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcp-project-id', type=str, required=True)
    parser.add_argument('--gcp-project-location', type=str, required=True)
    parser.add_argument('--target-movie-clip', type=str, default="assets/sample1.mp4")
    parser.add_argument('--prompt-tpl-path', type=str, default="vid2persona/prompts")
    args = parser.parse_args()

    workflow(args)
