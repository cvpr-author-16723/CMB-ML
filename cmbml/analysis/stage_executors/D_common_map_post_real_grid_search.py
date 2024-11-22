from typing import List, Dict
import logging
from itertools import product

from hydra.utils import instantiate
import numpy as np
from tqdm import tqdm
import healpy as hp

from omegaconf import DictConfig

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from cmbml.utils.physics_mask import downgrade_mask


logger = logging.getLogger(__name__)


class CommonPostExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str: str) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str)

        self.out_cmb_map_real: Asset = self.assets_out["cmb_map"]
        out_ps_handler: HealpyMap

        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        in_cmb_map_handler: HealpyMap

        # Basic parameters
        self.nside_out = self.cfg.scenario.nside
        self.lmax = int(cfg.model.analysis.lmax_ratio * self.nside_out)

        # Prepare to load mask (in execute())
        self.mask_threshold = self.cfg.model.analysis.mask_threshold

        self.use_pixel_weights = False

        # Prepare to load beam and mask in execute()
        self.beam = None
        self.mask = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.mask = self.get_masks()
        self.beam = self.get_beam()
        self.default_execute()

    def get_masks(self):
        mask = None
        with self.name_tracker.set_context("src_root", self.cfg.local_system.assets_dir):
            logger.info(f"Using mask from {self.in_mask.path}")
            mask = self.in_mask.read(map_fields=self.in_mask.use_fields)[0]
        mask = downgrade_mask(mask, self.nside_out, threshold=self.mask_threshold)
        return mask

    def get_beam(self):
        # Partially instantiate the beam object, defined in the hydra configs (cfg.model.analysis)
        beam = instantiate(self.beam_cfg)  # Defined in children classes (at bottom of file)
        beam = beam(lmax=self.lmax)
        return beam

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        
        epochs = self.model_epochs if self.model_epochs else [""]

        for epoch in epochs:
            for sim in tqdm(split.iter_sims()):
                context_params = dict(epoch=epoch, sim_num=sim)
                with self.name_tracker.set_contexts(context_params):
                    self.process_sim()

    def process_sim(self) -> None:
        # Get power spectrum for realization
        cmb_map: np.ndarray = self.in_cmb_map.read()
        if cmb_map.shape[0] == 3 and self.map_fields == "I":
            cmb_map = cmb_map[0]

        masking = ["hpma", "yxma", "noma"]
        deconv = [True, False]
        remove_dipole = [True, False]

        combos = list(product(masking, deconv, remove_dipole))
        for combo in combos:
            self.process_combo(cmb_map, *combo)


    def process_combo(self, cmb_map, masking, deconv, remove_dipole):
        if masking == "hpma":
            post_map = hp.ma(cmb_map)
            # healpy wants True for pixels to be masked; convention is False
            post_map.mask = np.logical_not(self.mask)
        elif masking == "yxma":
            post_map = cmb_map * self.mask
        else:
            post_map = cmb_map

        if deconv:
            post_map = self.deconv(post_map)

        if remove_dipole:
            post_map = hp.remove_dipole(post_map)

        context_params = dict(
            mask=masking,
            deconv="ydec" if deconv else "ndec",
            remove_dipole="yrd" if remove_dipole else "nrd"
        )
        with self.name_tracker.set_contexts(context_params):
            self.out_cmb_map_real.write(data=post_map)

    def deconv(self, data) -> np.ndarray:
        alm_in = hp.map2alm(data, lmax=self.lmax)
        alm_deconv = hp.almxfl(alm_in, 1 / self.beam.beam[:self.lmax])
        map_deconv = hp.alm2map(alm_deconv, nside=self.nside_out)
        return map_deconv


class CommonRealPostExecutor(CommonPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="common_post_map_real")
        self.out_cmb_map: Asset = self.assets_out["cmb_map"]
        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.beam_cfg = cfg.model.analysis.beam_real

    def deconv(self, data):
        """
        No-op because the realization map was never convolved
        """
        return data


class CommonCMBNNCSPredPostExecutor(CommonPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="common_post_map_pred")
        self.out_cmb_map: Asset = self.assets_out["cmb_map"]
        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.beam_cfg = cfg.model.analysis.beam_cmbnncs


class CommonPyILCPredPostExecutor(CommonPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="common_post_map_pred")
        self.out_cmb_map: Asset = self.assets_out["cmb_map"]
        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.beam_cfg = cfg.model.analysis.beam_pyilc
