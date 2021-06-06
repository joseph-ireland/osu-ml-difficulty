import click
from . import difficulty_model, replay_features, map_difficulty


@click.group()
def main():
    pass


@main.command()
@click.argument("filename", default="model.keras")
@click.option("--batch-count", "-b", default=None, type=int,
              help="Amount of data batches to train with (default: all)")
def train(filename, batch_count):
    """
    Train a model
    """
    difficulty_model.fit(filename, batch_count)


@main.command()
@click.option("--force/--no-force", "-f", default=False,
              help="Recalculate all features even if they already exist")
def extract_replay_features(force):
    """
    Extract hit information from replays. Required before running training
    """
    replay_features.calculate_all_replay_features(force=force)

@main.command()
@click.argument("map_csv", type=click.Path(exists=True))
@click.option("--model_path", type=click.Path(exists=True), default="model.keras",
              help="Path to model")
def map_list(map_csv, model_path):
    """
    map_csv:  Path to csv map list. Must contain 'ID' and 'Mods' columns"
    """

    map_difficulty.map_required_skills_from_csv(map_csv, model_path)



main()
