import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


class ScreenData:
    """
    Class for working with screen data, with configurable output paths.
    """

    def __init__(
        self,
        file_path: str,
        id_col: str,
        val_col: str,
        bio_taskname: str,
        base_dir: Path | str | None = None,
        id_: str = 'Gene',
        colname: str = 'Score',
        save: bool = True
    ):
        """
        Initialize ScreenData object.

        Args:
            file_path (str): Path to the data file.
            id_col (str): Name of the column with ID information for each gene.
            val_col (str): Name of the value column with screen readout.
            bio_taskname (str): Name of the bio task.
            base_dir (Path or str, optional): Base directory for saving outputs. Defaults to current working directory.
            id_ (str): New name for the id column.
            colname (str): New name for the value column.
            save (bool): Whether to save ground truth immediately.

        Raises:
            ValueError: If the file type is incorrect.
        """
        # Load data
        if file_path.endswith('.csv'):
            self.data_df = pd.read_csv(file_path)
        elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
            self.data_df = pd.read_csv(file_path, delimiter='\t')
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        # Rename and index
        self.data_df = (
            self.data_df
            .rename(columns={id_col: id_, val_col: colname})
            .set_index(id_)
            .loc[:, [colname]]
        )

        self.bio_taskname = bio_taskname
        self.colname = colname

        # Setup directories
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.datasets_dir = self.base_dir / 'datasets'
        self.prompts_dir = self.datasets_dir / 'task_prompts'
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        # Optionally save ground truth
        if save:
            self.save_ground_truth()

    def save_ground_truth(self) -> None:
        """
        Save the ground truth data to a CSV file in datasets directory.
        """
        out_path = self.datasets_dir / f"ground_truth_{self.bio_taskname}.csv"
        self.data_df.to_csv(out_path)

    def identify_hits(
        self,
        type_: str = 'gaussian',
        top_ratio_threshold: float = 0.05,
        save: bool = True
    ) -> None:
        """
        Identify hits in the data and optionally save indices of top movers.

        Args:
            type_ (str): Method for thresholding ('castle' or 'gaussian').
            top_ratio_threshold (float): Fraction of top absolute values to select (for 'gaussian').
            save (bool): Whether to save the list of top movers.
        """
        vals = self.data_df[self.colname].values

        if type_ == 'castle':
            plt.hist(vals, bins=100)
            plt.yscale('log')
            thresh = np.percentile(vals, 95)
            plt.axvline(x=thresh, color='r', linestyle='--')
            self.topmovers = list(self.data_df[self.data_df[self.colname] > thresh].index)

        elif type_ == 'gaussian':
            count = int(len(vals) * top_ratio_threshold)
            indices = np.argsort(np.abs(vals))[::-1][:count]
            self.topmovers = list(self.data_df.iloc[indices].index)

        else:
            raise ValueError(f"Unknown type_: {type_}")

        if save:
            out_path = self.datasets_dir / f"topmovers_{self.bio_taskname}.npy"
            np.save(out_path, np.array(self.topmovers, dtype=object))

    def set_task_prompt(self, task_description: str, measurement: str) -> None:
        """
        Write a JSON file containing the task prompt and measurement.
        """
        prompt_data = {
            'Task': task_description,
            'Measurement': measurement
        }
        out_path = self.prompts_dir / f"{self.bio_taskname}.json"
        with open(out_path, 'w') as f:
            json.dump(prompt_data, f, indent=2)


def read_task_prompt(json_file_path: str) -> tuple[str, str]:
    """
    Reads the task prompt from a JSON file and returns the contents.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data.get('Task', ''), data.get('Measurement', '')
