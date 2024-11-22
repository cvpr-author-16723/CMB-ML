from typing import List, Dict, Union
from typing import List, Dict, Union
import logging

import numpy as np

import matplotlib.pyplot as plt
from omegaconf import DictConfig
import healpy as hp

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
# from tqdm import tqdm
from cmbml.core.asset_handlers.asset_handlers_base import Mover
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.utils import planck_cmap


logger = logging.getLogger(__name__)


class ShowSimsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str="show_sims")

        self.out_cmb_figure: Asset = self.assets_out["cmb_map_render"]
        self.out_obs_figure: Asset = self.assets_out["obs_map_render"]
        out_cmb_figure_handler: Mover
        out_obs_figure_handler: Mover

        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.in_obs_map: Asset = self.assets_in["obs_maps"]
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap

        self.instrument: Instrument = make_instrument(cfg=cfg)

        # Only produce visualizations for a subset of sims
        self.sim_ns = self.get_override_sim_ns(cfg.pipeline[self.stage_str].override_n_sims)
        self.min_max = self.get_plot_min_max(cfg.pipeline[self.stage_str].plot_min_max)
        
        # For Gnomonic view
        self.plot_rot = cfg.pipeline[self.stage_str].plot_rot
        self.gnom_plot_res = cfg.pipeline[self.stage_str].plot_gnom_res

    def get_override_sim_ns(self, sim_nums: Union[None, list, int]):
        # Returns either a list of sims, or None
        try:
            return list(range(sim_nums))
        except TypeError:
            return sim_nums

    def get_plot_min_max(self, min_max):
        if min_max is None:
            plot_min = plot_max = None
        elif isinstance(min_max, int):
            plot_min = -min_max
            plot_max = min_max
        elif isinstance(min_max, list):
            plot_min = min_max[0]
            plot_max = min_max[1]
        return plot_min, plot_max

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.default_execute()

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")

        # We may want to process a subset of all sims
        if self.sim_ns is None:
            sim_iter = split.iter_sims()
        else:
            sim_iter = self.sim_ns

        for sim in sim_iter:
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        cmb_map = self.in_cmb_map.read()
        self.make_maps_per_field(cmb_map, det="cmb", out_asset=self.out_cmb_figure)
        for freq in self.instrument.dets:
            with self.name_tracker.set_context("freq", freq):
                obs_map = self.in_obs_map.read()
                self.make_maps_per_field(obs_map, det=freq, out_asset=self.out_obs_figure)

    def make_maps_per_field(self, some_map, det, out_asset):
        split = self.name_tracker.context['split']
        sim_n = f"{self.name_tracker.context['sim_num']:0{self.cfg.file_system.sim_str_num_digits}d}"
        if det == "cmb":
            title_start = "CMB Realization (Target)"
            fields = self.cfg.scenario.map_fields
        else:
            title_start = f"Observation, {det} GHz (Feature)"
            fields = self.instrument.dets[det].fields
        for field_str in fields:
            with self.name_tracker.set_context("field", field_str):
                field_idx = {'I': 0, 'Q': 1, 'U': 2}[field_str]
                # self.make_gnomview(some_map[field_idx])
                # with self.name_tracker.set_context("view", "gnom"):
                #     self.save_figure(title_start, split, sim_n, field_str, out_asset)
                self.make_mollview(some_map[field_idx])
                with self.name_tracker.set_context("view", "moll"):
                    self.save_figure(title_start, split, sim_n, field_str, out_asset)

    def save_figure(self, title, split_name, sim_num, field_str, out_asset):
        plt.suptitle(f"{title}, {split_name}:{sim_num} {field_str} Stokes")

        fn = out_asset.path.name
        plt.savefig(fn)
        plt.close()
        out_asset.write(source_location=fn)

    def make_mollview(self, some_map):
        fig = plt.figure(figsize=(8, 6))
        plot_params = dict(
            min=self.min_max[0], 
            max=self.min_max[1],
            rot=self.plot_rot,
            cmap=planck_cmap.colombi1_cmap,
            hold=True,
            norm='log'
        )
        hp.mollview(some_map, **plot_params)

    def make_gnomview(self, some_map):
        fig = plt.figure(figsize=(8, 6))
        plot_params = dict(
            min=self.min_max[0], 
            max=self.min_max[1],
            rot=self.plot_rot,
            reso=self.gnom_plot_res,
            cmap=planck_cmap.colombi1_cmap,
            hold=True,
            norm='log'
        )
        hp.gnomview(some_map, **plot_params)
