import argparse
import os
import logging
from pathlib import Path
import sys
import torch
from config import config
from training.train import SparseGraphTrainer

# Configure CUDA for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_logging():
    """Configure logging with both console and file output"""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_DIR / 'training.log')
        ]
    )
    return logging.getLogger(__name__)

def setup_environment():
    """Configure system environment for the project"""
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    
    # Optimize CUDA performance
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

def parse_arguments():
    """Parse command line arguments with enhanced validation"""
    parser = argparse.ArgumentParser(
        description="Hierarchical GNN for PanNuke Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mode', type=str, required=True,
                      choices=['train', 'evaluate', 'explain', 'validate', 'generate_graphs'],
                      help="Operation mode")
    parser.add_argument('--model-path', type=str, 
                      default=str(config.MODEL_SAVE_PATH / 'best_model.pth'),
                      help="Path to model checkpoint")
    parser.add_argument('--num-workers', type=int, 
                      default=min(4, os.cpu_count() - 1) if os.cpu_count() > 1 else 0,
                      help="Number of data loading workers")
    parser.add_argument('--debug', action='store_true',
                      help="Enable debug mode")
    parser.add_argument('--fold', type=int, default=0,
                      help="Fold number for cross-validation (0-4)")
    return parser.parse_args()

def validate_paths():
    """Ensure all required directories exist"""
    required_dirs = [
        config.LOG_DIR,
        config.MODEL_SAVE_PATH,
        config.RESULTS_PATH,
        config.GRAPH_DATA_PATH
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

def main():
    """Main execution function with comprehensive error handling"""
    # Initial setup
    setup_environment()
    global logger
    logger = setup_logging()
    args = parse_arguments()
    validate_paths()

    try:
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
            torch.autograd.set_detect_anomaly(True)

        # Execute the requested operation
        if args.mode == 'train':
            logger.info("Initializing training pipeline")
            trainer = SparseGraphTrainer(config)
            trainer.train()
        elif args.mode == 'evaluate':
            logger.info("Initializing evaluation pipeline")
            from training.evaluate import Evaluator
            evaluator = Evaluator(args.model_path)
            evaluator.evaluate()
        elif args.mode == 'explain':
            logger.info("Initializing explainability pipeline")
            from training.explainability import explain_model_on_dataset
            explain_model_on_dataset(args.model_path)
        elif args.mode == 'validate':
            logger.info("Initializing external validation")
            from training.external_validation import run_external_validations
            run_external_validations(args.model_path)
        elif args.mode == 'generate_graphs':
            logger.info("Generating graphs from dataset")
            from utils.graph_generator import main as generate_graphs
            generate_graphs()

    except Exception as e:
        logger.error(f"Fatal error during {args.mode} operation: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up CUDA resources
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    logger.info("Operation completed successfully")

if __name__ == "__main__":
    main()
