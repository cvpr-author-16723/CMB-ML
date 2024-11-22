from typing import List, Dict
import logging

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
# from src.analysis.make_ps import get_power as _get_power
from cmbml.core.asset_handlers.psmaker_handler import NumpyPowerSpectrum
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from cmbml.utils.physics_ps import get_auto_ps_result, get_x_ps_result, PowerSpectrum
from cmbml.utils.physics_beam import NoBeam, GaussianBeam
from cmbml.utils.physics_mask import downgrade_mask

# import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class MakePredPowerSpectrumExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, beam_type:str) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="make_pred_ps")

        self.out_auto_real: Asset = self.assets_out.get("auto_real", None)
        self.out_auto_pred: Asset = self.assets_out.get("auto_pred", None)
        # self.out_x_real_pred: Asset = self.assets_out.get("x_real_pred", None)
        out_ps_handler: NumpyPowerSpectrum

        self.in_cmb_map_real: Asset = self.assets_in["cmb_map_real"]
        self.in_cmb_map_pred: Asset = self.assets_in["cmb_map_post"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        self.in_mask_sm: Asset = self.assets_in.get("mask_sm", None)
        in_cmb_map_handler: HealpyMap

        # Basic parameters
        self.nside_out = self.cfg.scenario.nside
        self.lmax = int(cfg.model.analysis.lmax_ratio * self.nside_out)

        # Prepare to load mask (in execute())
        self.mask_threshold = self.cfg.model.analysis.mask_threshold
        self.mask = None

        self.use_sm_mask = self.cfg.model.analysis.ps_use_smooth_mask

        # Prepare to load beam (in execute())
        # beam_type is either "beam_pyilc" or "beam_other"
        self.beam_real = None
        self.beam_pred = cfg.model.analysis.get(beam_type, None)

        self.use_pixel_weights = False

        if self.cfg.map_fields != "I":
            raise NotImplementedError("Only intensity maps are currently supported.")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.mask = self.get_masks()
        # self.beam_real = GaussianBeam(beam_fwhm=5, lmax=self.lmax)
        self.beam_real = NoBeam(self.lmax)
        self.beam_pred = self.get_pred_beam()
        self.default_execute()

    def get_masks(self):
        mask = None
        with self.name_tracker.set_context("src_root", self.cfg.local_system.assets_dir):
            logger.info(f"Using mask from {self.in_mask.path}")
            if self.use_sm_mask:
                mask = self.in_mask_sm.read(map_fields=self.in_mask_sm.use_fields)[0]
            else:
                mask = self.in_mask.read(map_fields=self.in_mask.use_fields)[0]
        if hp.npix2nside(mask.size) != self.nside_out:
            mask = downgrade_mask(mask, self.nside_out, threshold=self.mask_threshold)
        return mask

    def get_pred_beam(self):
        # Partially instantiate the beam object, defined in the hydra configs
        # Currently tested are GaussianBeam and NoBeam, which differ only in how they are instantiated
        beam = instantiate(self.beam_pred)
        beam = beam(lmax=self.lmax)
        return beam

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        for sim in tqdm(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        # Get power spectrum for realization
        real_map: np.ndarray = self.in_cmb_map_real.read()
        if real_map.shape[0] == 3 and self.map_fields == "I":
            real_map = real_map[0]
        self.make_real_ps(real_map)

        # Get power spectra for predictions
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                # We may want to generate cross power spectra as well
                # TODO: Make flag for this in config file instead of hardcoding
                self.make_pred_ps(real_map)

    def make_real_ps(self, real_map):
        auto_real_ps = get_auto_ps_result(real_map,
                                          mask=None,
                                          lmax=self.lmax,
                                          beam=self.beam_real,
                                          is_convolved=False)
        # ps1 = auto_real_ps._ps
        ps = auto_real_ps.deconv_dl
        # print(max(ps-ps1))
        # TODO: Pixel Window handling? Why is Realization PS slightly low? At what stage in the pipeline should it be implemented correctly?
        # pix_win_512 = hp.pixwin(self.nside_out)
        # pix_win_2048 = hp.pixwin(2048)
        # pix_win_scale = (pix_win_512[:ps.size]) ** -2
        # pix_win_scale = (1 / pix_win_2048[:ps.size]) ** -2
        # pix_win_scale = (pix_win_512[:ps.size] / pix_win_2048[:ps.size]) ** -2
        self.out_auto_real.write(data=ps.value)

    def make_pred_ps(self, real_map) -> None:
        pred_map = self.in_cmb_map_pred.read()  # This is just the post-processed map, not the Common-Processed map
        auto_pred_ps = get_auto_ps_result(pred_map,
                                          mask=self.mask,
                                          lmax=self.lmax,
                                          beam=self.beam_pred,
                                          is_convolved=True)
        # ps1 = auto_pred_ps._ps
        ps = auto_pred_ps.deconv_dl
        pix_win_scale = 1
        # TODO: Pixel Window handling? Why is Realization PS slightly low? At what stage in the pipeline should it be implemented correctly?
        # pix_win_512 = hp.pixwin(self.nside_out)
        # pix_win_2048 = hp.pixwin(2048)
        # pix_win_scale = (pix_win_512[:ps.size]) ** -2
        # pix_win_scale = (1 / pix_win_2048[:ps.size]) ** -2
        # pix_win_scale = (pix_win_512[:ps.size] / pix_win_2048[:ps.size]) ** -2
        ps = ps * pix_win_scale
        # print(max(ps-ps1))
        self.out_auto_pred.write(data=ps.value)


class PyILCMakePSExecutor(MakePredPowerSpectrumExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, "beam_pyilc")


class CMBNNCSMakePSExecutor(MakePredPowerSpectrumExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, "beam_cmbnncs")
