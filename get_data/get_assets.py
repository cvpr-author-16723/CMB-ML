import logging
import hydra
from cmbml.core import PipelineContext, LogMaker
from get_data.stage_executors.A_get_assets import GetAssetsExecutor


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../cfg", config_name="config_sim")
def main(cfg):
    """
    Runs the simulation pipeline.

    Args:
        cfg: The configuration object.

    Raises:
        Exception: If an exception occurs during the pipeline execution.
    """
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(GetAssetsExecutor)

    pipeline_context.prerun_pipeline()

    had_exception = False
    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        had_exception = True
        logger.exception("An exception occurred during the pipeline.", exc_info=e)
        raise e
    finally:
        if had_exception:
            logger.error("Pipeline failed.")
        else:
            logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    main()
