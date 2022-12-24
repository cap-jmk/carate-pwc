import click

from carate.run import Run


@click.command()
@click.option("-c", help="Path to config file")
def train_algorithm(c: str, o: str) -> None:

    config_filepath = c
    runner = Run.from_file(config_filepath=config_filepath)
    runner.run()


if __name__ == "__main__":
    train_algorithm()
