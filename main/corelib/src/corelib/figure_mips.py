import logging
import os
from typing import Iterator, List, Optional
from basic_typing import Batch
import numpy as np
import itertools
from .figure_gallery import gallery
import matplotlib.pyplot as plt
from sequence import Sequence


def find_root_sequence(seq: Sequence) -> Sequence:
    if hasattr(seq, 'source_split') and seq.source_split is not None:
        return find_root_sequence(seq.source_split)
    return seq

def get_data_inmemory_fn(split: Sequence) -> Iterator[Batch]:
    root_sequence = find_root_sequence(split)
    assert isinstance(root_sequence, sequence_array.SequenceArray)
    return root_sequence


def compare_volumes_mips(
        volumes: List[List[np.ndarray]], 
        case_names: List[str], 
        category_names: List[str], 
        cmap=plt.get_cmap('plasma'),
        title: Optional[str] = None,
        dpi: Optional[int] = None,
        min_value: float = 0.0,
        max_value: Optional[float] = None,
        overlay_with: Optional[List[Optional[int]]] = None,
        fontsize: int = 12,
        with_xz=True,
        with_yz=True,
        with_xy=True,
        flip=False,
        mip_views_by_row=True,
        figsize=(16, 6)):
    """
    Generate Maximum intensity projections of volumes

    Args:
        volumes: a list of list of volumes, indexed by [case_id][category_id]
        case_names: the names of the cases 
        category_names: the names of the categories
        overlay_with: if define a base overlay for each volume (e.g., for segmentation, we want to
            show the base volume to check the boundaries)

    Returns:
        a figure
    """
    def transform(im, max_value, min_value):
        # make sure we have EXACTLY the same
        # intensity range so that the images
        # are easily comparable
        im = cmap(im / max_value)
        if flip:
            im = np.flipud(im)
        im = np.clip(im, min_value, 1.0)
        im = np.uint8(im * 255)
        return im


    export_mip_path = os.environ.get('INFERENCE_EXPORT_MIP_PATH')

    # check the sizes
    assert len(case_names) == len(volumes)
    for vs in volumes:
        assert len(vs) == len(category_names), f'expected #volumes={len(category_names)}, got={len(vs)}'
    
    y_axis_text = []
    rows = []
    for row_id, volumes_row in enumerate(volumes):
        if overlay_with is not None:
            assert len(volumes_row) == len(overlay_with)
        volume_rows = [[], [], []]

        # scale to max value so that all
        # volumes MIPs are comparable
        if max_value is None:
            current_max_value = volumes_row[0].max()
        else:
            current_max_value = max_value

        for v in volumes_row:
            if with_xz:
                mip_x_z = v.max(axis=1)
                volume_rows[0].append(transform(mip_x_z, max_value=current_max_value, min_value=min_value))
                y_axis_text.append(f'case={case_names[row_id]}, Mip XZ')

            if with_yz:
                mip_y_z = v.max(axis=2)
                volume_rows[1].append(transform(mip_y_z, max_value=current_max_value, min_value=min_value))
                y_axis_text.append(f'case={case_names[row_id]}, Mip YZ')
            
            if with_xy:
                mip_x_y = v.max(axis=0)
                volume_rows[2].append(transform(mip_x_y, max_value=current_max_value, min_value=min_value))
                y_axis_text.append(f'case={case_names[row_id]}, Mip XY')

        # discard empty rows
        volume_rows = [r for r in volume_rows if len(r) > 0]

        if export_mip_path:
            case_name = case_names[row_id]
            from PIL import Image
            import PIL
            
            def get_image(row_id, volume_n):
                if overlay_with is None or overlay_with[volume_n] is None:
                    return volume_rows[row_id][volume_n]
                else:
                    import copy
                    base_n = overlay_with[volume_n]
                    base = copy.deepcopy(volume_rows[row_id][base_n])
                    seg = volume_rows[row_id][volume_n]
                    seg_indices = np.where(np.all(seg == (0, 0, 0, 255), axis=-1))
                    base[seg_indices] = (0, 255, 0, 255)
                    return base

            v0 = np.concatenate([get_image(0, 0), get_image(0, 1), get_image(0, 2)], axis=1)
            v1 = np.concatenate([get_image(1, 0), get_image(1, 1), get_image(1, 2)], axis=1)
            #v0 = np.concatenate([get_image(row_id + 0, 0), get_image(row_id + 0, 1), get_image(row_id + 0, 2)], axis=1)
            #v1 = np.concatenate([get_image(row_id + 1, 0), get_image(row_id + 1, 1), get_image(row_id + 1, 2)], axis=1)
            #v0 = np.concatenate([volume_rows[row_id + 0][0], volume_rows[row_id + 0][1], volume_rows[row_id + 0][2]], axis=1)
            #v1 = np.concatenate([volume_rows[row_id + 1][0], volume_rows[row_id + 1][1], volume_rows[row_id + 1][2]], axis=1)
            xz = np.concatenate([v0, v1], axis=0)
            half_y = xz.shape[0] // 2
            third_x = xz.shape[1] // 3
            xz[half_y:half_y+1, :, 0:3] = 0
            xz[:, third_x:third_x+1, 0:3] = 0
            xz[:, 2*third_x:2*third_x+1, 0:3] = 0
            i = Image.fromarray(xz)
            i = i.resize((2 * xz.shape[1], 2 * xz.shape[0]), resample=PIL.Image.Resampling.BICUBIC)
            i.save(f'{export_mip_path}/{case_name}.png')

        if mip_views_by_row:
            # views will be on a single row
            rows.append(list(itertools.chain(*volume_rows)))
        else:
            # one additional row per mip view
            rows += volume_rows

    if mip_views_by_row:
        y_axis_text = case_names
        category_names = category_names * (int(with_xz) + int(with_xy) + int(with_yz))

    fig = gallery(
        images_y_then_x=rows,
        x_axis_text=category_names,
        y_axis_text=y_axis_text,
        title=title,
        save_path=None,
        dpi=dpi,
        fontsize=fontsize,
        figsize=figsize
    )

    return fig