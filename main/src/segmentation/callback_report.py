"""
Write a report that summarizes the current run and compare it
to other runs
"""
import itertools
import os
import shutil
from typing import Dict, Tuple, Optional
from callbacks.callback import Callback
import logging
import json
from mdutils.mdutils import MdUtils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import wilcoxon
from corelib import plot_scatter
import seaborn as sns
from corelib import compare_volumes_mips
import h5py
from options import Options
from utilities import create_or_recreate_folder, make_unique_colors_f


logger = logging.getLogger(__name__)

here = os.path.abspath(os.path.dirname(__file__))


def read_metrics(path: str) -> Dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def read_hdf5(path: str) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        v = f['v'][()]
    return v


def get_related_configurations(
        options: Options, 
        level=1,
        experiment_repository=('experiments', 'baseline', 'results')) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Load related experiment metrics & folder
    This is done by checking the `configuration.tracking['derived_from']` tag
    of the experiments. Multiple reference configurations can be specified using
    seperator `;`

    Args:
        options: options of the current run
        level: How deep do we go in the run depency to collect 
            related experiments
        experiment_repository: where the experiment results are saved
    Returns:
        a tuple of (Dictionary of (name, metrics), dictionary of (config name, config path)
    """
    related_configurations = []
    assert level == 1, 'TODO implement hierarchical configurations'
    derived_from = options.runtime.configuration.tracking.get('derived_from')
    if derived_from is not None:
        related_configurations = derived_from.split(';')
        
    experiments_metrics = {}
    experiments_path = {}
    for config_name in related_configurations:
        path = os.path.join(here, *experiment_repository, config_name.replace('.py', '.json'))
        if not os.path.exists(path):
            print(f'warning: cannot find results for run={config_name}, path={path}')
            logger.warning(f'cannot find results for run={config_name}, path={path}')
            continue

        metrics = read_metrics(path)
        experiments_metrics[config_name] = metrics
        experiment_path = os.path.join(
            options.runtime.configuration.training['logging_directory'], 
            metrics['experiment_name']
        )

        if os.path.exists(experiment_path):
            experiments_path[config_name] = experiment_path
        else:
            logger.warning('experiment folder coult not be found! path={experiment_path}')

    return experiments_metrics, experiments_path


def get_case_by_case_metric(metrics, metric_name, output_name, split_name) -> Optional[pd.DataFrame]:
    """
    Return the metrics by sample of a given split.
    Collect `case_uid` so that we can compare runs that were not performed
    on the same data.
    """
    raw_data = metrics.get('raw_data')
    if raw_data is None:
        return None
    
    metric_values = raw_data.get(metric_name)
    if metric_values is None:
        return None

    metric_values = np.asarray(metric_values, dtype=np.float32)
    uids = raw_data.get('case_uid')
    split = raw_data.get('dataset_split')

    included = np.asarray([split_name == s for s in split])
    return pd.DataFrame({'case_uid': np.asarray(uids)[included], output_name: metric_values[included]})


def get_metric_pairs_p_value(metrics_current, metrics_reference, metric_name, split_name) -> Optional[float]:
    """
    Calculate the p-value of paired measurements testing if the two distribution
    (current, reference) are the same.
    This is done using non-parametric paired test wilcoxon signed rank test.
    """
    metrics_current_values = get_case_by_case_metric(metrics_current, metric_name, 'current', split_name)
    metrics_reference_values = get_case_by_case_metric(metrics_reference, metric_name, 'ref', split_name)
    if metrics_current_values is None or metrics_reference_values is None:
        return None

    # they may be evaluated on a different subset, use the UIDs to find the common
    # test subset
    merged = pd.merge(metrics_current_values, metrics_reference_values, on='case_uid').dropna()

    nb_samples = len(merged)
    try:
        p_value = wilcoxon(x=merged['current'], y=merged['ref']).pvalue
    except Exception as e:
        logger.error(f'Exception={e} in <get_metric_pairs_p_value>')
        # invalid
        p_value = None
    return p_value, nb_samples


def format_number(number) -> str:
    return f'{number:.3f}'

        
def format_metrics_table(md, metrics_current, experiments_metrics, with_TN=True) -> None:
    """
    Create the main metrics table
    """
    aggregated_metrics = metrics_current['aggregated']
    experiments_comparison = list(experiments_metrics.keys())
    lines = [['metric', 'dataset', 'current'] + experiments_comparison]
    for metric_name, kvp in aggregated_metrics.items():
        for name, value in kvp.items():
            line = [metric_name, name, format_number(value)]
            for e in experiments_comparison:
                metric_e = experiments_metrics[e]['aggregated'].get(metric_name)
                if metric_e is None:
                    line.append('N/A')
                else:
                    value_e = metric_e.get(name)
                    if value_e is None:
                        line.append('N/A')
                    else:
                        p_value, nb_samples = get_metric_pairs_p_value(metrics_current, experiments_metrics[e], metric_name, name)
                        s = format_number(value_e)
                        if p_value is not None:
                            s += f' (p={format_number(p_value)}, N={nb_samples})' 
                        line.append(s)
            lines.append(line)

    if with_TN:
        # True negative cases is an important metric to capture to measure the FP counts
        def get_TN(metrics):
            dice = metrics['raw_data']['dice']
            df = pd.DataFrame({'dice': dice})
            return sum(df.isna().values).item()

        line = ['TN', 'All', get_TN(metrics_current)]
        for e in experiments_comparison:
            nb_tn = get_TN(experiments_metrics[e])
            line.append(nb_tn)
        lines.append(line)

    md.new_table(columns=len(lines[0]), rows=len(lines), text=list(itertools.chain(*lines)), text_align='left')
    

    md.new_line('Metrics definition:')
    md.new_list([
        'recon_time: the time to do a whole-body inference',
        'dice: the average dice (per patient) as calculated by the AutoPET challenge. In particular, a FP without foreground => dice = 0, no foreground and no segmentation found => case is skipped',
        'dice_foreground: the dice is ONLY calculated for the cases with foreground',
        'false_neg_vol: the volume of the false negative lesions',
        'false_pos_vol: the volume of the false positive lesions',
        'TN: the number of cases that correctly found NO lesion'
    ])


def format_worst_examples(md, metric_ref, metrics_current, current_experiment_path, output_path_report, nb_examples, basename, cmap, metrics_0_is_best) -> None:
    """
    Create the report section that extract the worst examples for a given metric
    """
    df = pd.DataFrame(metrics_current['raw_data'])
    last_epoch = metrics_current['epoch']
    if metric_ref not in df:
        md.new_line('The metric is missing!')
        return

    ascending = True
    if metric_ref in metrics_0_is_best:
        # we need to revert the sorting! 0 = best, 1 = worst
        ascending = False

    df = df.sort_values(metric_ref, ascending=ascending).head(nb_examples)
    
    volumes_by_row = []
    metric_values = df[metric_ref]
    lines = [['case_uid', 'metric']]
    for n in range(len(df)):
        try:
            dataset = df['dataset_split'].iloc[n].replace('/', '_')
            uid = df['case_uid'].iloc[n]
            value = df[metric_ref].iloc[n]
            filename_found = f'{dataset}_{uid}{basename}output_found.hdf5'
            filename_target = f'{dataset}_{uid}{basename}output_truth.hdf5'
            filename_input = f'{dataset}_{uid}{basename}output_input.hdf5'
            path_found = os.path.join(current_experiment_path, filename_found)
            path_target = os.path.join(current_experiment_path, filename_target)
            path_input = os.path.join(current_experiment_path, filename_input)

            found_np = read_hdf5(path_found)
            target_np = read_hdf5(path_target)
            input_np = read_hdf5(path_input)
            volumes_by_row.append([input_np, found_np, target_np])
            
            lines.append([uid, format_number(value)])
        except Exception as e:
            logger.error(f'case={uid} failed to load. Exception={e}' )

    if len(lines) == 1:
        # could not find the daa, discard this callback
        logger.info('No data could be found! Aborting=<format_worst_examples>')
        return

    # display all MIPs in a single figure to save space
    fig = compare_volumes_mips(
        volumes=volumes_by_row,
        case_names=[format_number(v) for v in metric_values.values],
        category_names=['input', 'found', 'target'],
        cmap=cmap,
        title=f'Worst case `{metric_ref}`',
        max_value=0.2,
        with_xy=False,
        figsize=(6, 12),
        flip=True,
        dpi=200
    )

    filename = f'worst_{metric_ref}.png'
    output_path = os.path.join(output_path_report, filename)
    fig.tight_layout()
    fig.savefig(output_path)

    md_image = md.new_inline_image(path=filename, text='')
    md.new_line(md_image)

    md.new_line('Worst cases info:')
    md.new_table(columns=len(lines[0]), rows=len(lines), text=list(itertools.chain(*lines)), text_align='left')


def format_histogram_comparison(md, metric_name, metrics_refs_dict, metrics_current, output_path):
    df_current = pd.DataFrame(metrics_current['raw_data'])
    metrics_pd = pd.DataFrame({
            metric_name: df_current[metric_name],
            'dataset_split': df_current['dataset_split'],
            'experiment': 'current'
        })

    # vertically concatenate the experiments
    for ref_name, ref_metrics in metrics_refs_dict.items():
        pd_ref = pd.DataFrame({
            metric_name: ref_metrics['raw_data'][metric_name],
            'dataset_split': ref_metrics['raw_data']['dataset_split'],
            'experiment': ref_name
        })

        metrics_pd = metrics_pd.append(pd_ref, ignore_index=True)

    # remove the NaN (no segmentation found & none expected)
    is_dice = 'dice' == metric_name
    fig, axes = plt.subplots(1, 1 + is_dice)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    fig.suptitle(f'Histogram for metric `{metric_name}`')
    sns.histplot(ax=axes[0], data=metrics_pd.dropna(), x=metric_name, hue='experiment', multiple='dodge', shrink=0.8, bins=10)
    sns.move_legend(
        axes[0], "lower center",
        bbox_to_anchor=(.5, 1),
        frameon=False,
    )
    
    if is_dice:
        # important metric for Dice: 50% of the scan don't have lesions
        # so we want to know how often false positive happen
        # by displaying the true negative cases
        is_nan = metrics_pd[metric_name].isna()
        metrics_pd_nan = metrics_pd[is_nan]
        metrics_pd_nan[metric_name] = 'TN'  # NaN are dropped so replace by dummy value
        sns.histplot(ax=axes[1], data=metrics_pd_nan, x=metric_name, hue='experiment', multiple='dodge', shrink=0.8)
        
        sns.move_legend(
            axes[1], "lower center",
            bbox_to_anchor=(.5, 1),
            frameon=False,
        )

    fig.tight_layout()
    fig.savefig(output_path)

    md_image = md.new_inline_image(
        text=f'Histogram ({metric_name})',
        path=os.path.basename(output_path)
    )
    md.new_line(md_image)


def format_scatter_comparison(md, experiment_name, metric_name, metrics_ref, metrics_current, output_path):
    df_current = pd.DataFrame(metrics_current['raw_data'])
    df_ref = pd.DataFrame(metrics_ref['raw_data'])
    if metric_name in df_current and metric_name in df_ref:

        left = pd.DataFrame({
            'case_uid': df_current['case_uid'],
            'current': df_current[metric_name],
            'dataset_split': df_current['dataset_split']
        })

        right = pd.DataFrame({
            'case_uid': df_ref['case_uid'],
            'ref': df_ref[metric_name],
        })

        merged = pd.merge(left, right, on='case_uid')
        if len(merged) > 0:
            split = [s.split('/')[1] for s in merged['dataset_split'].values]
            path = os.path.join(output_path, f'scatter_{metric_name}.png')
            merged_no_na = merged.dropna()
            robust_fit = len(merged_no_na) > 5
            plot_scatter(
                f'Scatter ({metric_name})',
                x=merged_no_na['current'].values,
                y=merged_no_na['ref'].values,
                xlabel='current',
                ylabel=experiment_name,
                path=path,
                label_class=split,
                colors=make_unique_colors_f(),
                xy_label=list(merged_no_na['case_uid'].values),
                robust_linear_fit=robust_fit
            )

            md_image = md.new_inline_image(
                text=f'Scatter ({metric_name})',
                path=os.path.basename(path)
            )
            md.new_line(md_image)
        else:
            md.new_line('No comparison available!')
    else:
        md.new_line('No comparison available!')


class CallbackWriteReport(Callback):
    """
    Write a report that can directly be added to the github repository
    README.md so that other shareholders can evaluate the results
    of experiments in a transparent manner
    """
    def __init__(
            self, 
            output_dir_name='experiments/baseline/results/report', 
            splits_to_consider=('test', 'valid'), 
            metrics_to_display=('dice', 'dice_foreground', 'false_neg_vol', 'false_pos_vol'),
            metrics_worst_case_skip=('dice'),
            metrics_0_is_best=('false_neg_vol', 'false_pos_vol'),
            cmap=plt.get_cmap('binary'),
            nb_examples=10,
            experiment_repository=('experiments', 'baseline', 'results')):
        super().__init__()
        self.output_dir_name = output_dir_name
        self.output_path_local = None
        self.splits_to_consider = splits_to_consider
        self.metrics_to_display = metrics_to_display
        self.nb_examples = nb_examples
        self.experiment_repository = experiment_repository
        self.metrics_worst_case_skip = metrics_worst_case_skip
        self.cmap = cmap
        self.metrics_0_is_best = metrics_0_is_best

    def __call__(self, options, history, model, **kwargs):
        logger.info('CallbackWriteReport started')
        current_experiment_path = options.workflow_options.current_logging_directory
        if self.output_path_local is None:
            self.output_path_local = os.path.join(current_experiment_path, 'report')
            create_or_recreate_folder(self.output_path_local)

        current_metrics_path = os.path.join(current_experiment_path, 'metrics.json')
        if not os.path.exists(current_metrics_path):
            logger.error(f'metrics file does not exist! path={current_metrics_path}')
            return
        metrics_current = read_metrics(current_metrics_path)
        experiments_metrics, experiments_path = get_related_configurations(
            options, 
            experiment_repository=self.experiment_repository
        )
        
        base_level = 1
        md = MdUtils(file_name=os.path.join(self.output_path_local, 'results.md'),title='Results') 
        md.new_line(f'Configuration tested: {os.path.basename(sys.argv[0])}')   
        md.new_header(level=base_level, title='Metrics')
        md.new_line('Metrics are reported for each dataset and configuration.' \
                    'p-value `p` is calculated using Wilcoxon signed-rank test.')
        format_metrics_table(md, metrics_current, experiments_metrics)
        
        md.new_header(level=base_level, title='Comparison with references')
        md.new_line('This section will compare the current experiment against all the selected baselines (histogram).')
        for metric_name in self.metrics_to_display:
            output_path = os.path.join(self.output_path_local, f'hist_{metric_name}.png')
            format_histogram_comparison(md, metric_name, experiments_metrics, metrics_current, output_path)

        md.new_line('This section will compare the current experiment against all the selected baselines (scatter).')
        for experiment_name, metrics_ref in experiments_metrics.items():
            for metric_name in self.metrics_to_display:
                md.new_header(level=base_level + 1, title=f'{metric_name}/{experiment_name}')
                metrics_ref = experiments_metrics[experiment_name]
                format_scatter_comparison(md, experiment_name, metric_name, metrics_ref, metrics_current, self.output_path_local)

        md.new_header(level=base_level, title=f'Examples')
        for metric_name in self.metrics_to_display:
            if metric_name in self.metrics_worst_case_skip:
                continue

            md.new_header(level=base_level + 1, title=f'Worst cases (metric={metric_name})')
            format_worst_examples(
                md, 
                metric_name, 
                metrics_current, 
                os.path.join(current_experiment_path, 'wholebody_inference'), 
                self.output_path_local, 
                self.nb_examples,
                basename='',
                cmap=self.cmap,
                metrics_0_is_best=self.metrics_0_is_best,
            )

        if 'mtv_found' in metrics_current['raw_data']:
            md.new_header(level=base_level, title=f'MTV')
            mtv_found = metrics_current['raw_data']['mtv_found']
            mtv_truth = metrics_current['raw_data']['mtv_truth']

            filename = 'mtv.png'
            mtv_path = os.path.join(self.output_path_local, filename)
            
            plot_scatter(
                f'Scatter Metabolic Tumor Volume (MTV)',
                x=np.asarray(mtv_found),
                y=np.asarray(mtv_truth),
                xlabel='MTV Found',
                ylabel='MTV Truth',
                path=mtv_path,
                label_class=['p'],
                colors=make_unique_colors_f(),
                robust_linear_fit=False
            )

            md_image = md.new_inline_image(path=filename, text='')
            md.new_line(md_image)

        md.create_md_file()

        if self.output_dir_name is not None:
            output_path = os.path.join(here, self.output_dir_name)
            create_or_recreate_folder(output_path)
            shutil.copytree(self.output_path_local, output_path, dirs_exist_ok=True)

        logger.info('CallbackWriteReport done!')