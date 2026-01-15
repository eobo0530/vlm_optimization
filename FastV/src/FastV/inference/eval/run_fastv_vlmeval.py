#!/usr/bin/env python3
"""
FastV Evaluation Script using VLMEvalKit

This script evaluates FastV-enabled LLaVA models on MMBench and COCO datasets
using the VLMEvalKit framework.

Usage:
    python run_fastv_vlmeval.py --model-path llava-hf/llava-1.5-7b-hf --data MMBench_DEV_EN_V11 COCO_VAL --use-fast-v
    python run_fastv_vlmeval.py --config config.json
"""

import argparse
import json
import sys
import os
import os.path as osp

# Add paths
VLMEVAL_PATH = "/home/aips/VLMEvalKit"
FASTV_SRC_PATH = osp.join(osp.dirname(__file__), "../../..")
sys.path.insert(0, FASTV_SRC_PATH)
sys.path.insert(0, VLMEVAL_PATH)

# Import minimal VLMEvalKit components
from vlmeval.smp.log import get_logger
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
import pandas as pd

from fastv_wrapper import FastVLLaVA


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate FastV with VLMEvalKit on MMBench and COCO'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the LLaVA model (e.g., llava-hf/llava-1.5-7b-hf)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Name for the model (default: use model path basename)'
    )
    
    # FastV configuration
    parser.add_argument(
        '--use-fast-v',
        action='store_true',
        default=False,
        help='Enable FastV token pruning'
    )
    parser.add_argument(
        '--fast-v-inplace',
        action='store_true',
        default=False,
        help='Use inplace FastV for latency measurement'
    )
    parser.add_argument(
        '--fast-v-sys-length',
        type=int,
        default=35,
        help='Length of system prompt tokens (default: 35)'
    )
    parser.add_argument(
        '--fast-v-image-token-length',
        type=int,
        default=576,
        help='Number of image tokens (default: 576 for 24x24 patches)'
    )
    parser.add_argument(
        '--fast-v-attention-rank',
        type=int,
        default=100,
        help='Number of top attention tokens to keep (default: 100)'
    )
    parser.add_argument(
        '--fast-v-agg-layer',
        type=int,
        default=3,
        help='Layer index at which to aggregate attention scores (default: 3)'
    )
    
    # Dataset configuration
    parser.add_argument(
        '--data',
        type=str,
        nargs='+',
        default=['MMBench_DEV_EN_V11', 'COCO_VAL'],
        help='Dataset names to evaluate (default: MMBench_DEV_EN_V11 COCO_VAL)'
    )
    
    # Evaluation configuration
    parser.add_argument(
        '--work-dir',
        type=str,
        default='./outputs_fastv',
        help='Output directory for results (default: ./outputs_fastv)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'infer', 'eval'],
        help='Evaluation mode: all (infer+eval), infer only, or eval only (default: all)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--reuse',
        action='store_true',
        help='Reuse existing prediction files'
    )
    
    # Alternative: JSON config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON configuration file (overrides other arguments)'
    )
    
    return parser.parse_args()


def load_config_from_json(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def build_fastv_model(args):
    """Build FastV-enabled LLaVA model."""
    model_kwargs = {
        'use_fast_v': args.use_fast_v,
        'fast_v_inplace': args.fast_v_inplace,
        'fast_v_sys_length': args.fast_v_sys_length,
        'fast_v_image_token_length': args.fast_v_image_token_length,
        'fast_v_attention_rank': args.fast_v_attention_rank,
        'fast_v_agg_layer': args.fast_v_agg_layer,
    }
    
    logger = get_logger('BUILD_MODEL')
    logger.info(f"Building FastV model from {args.model_path}")
    logger.info(f"FastV configuration: {model_kwargs}")
    
    model = FastVLLaVA(model_path=args.model_path, **model_kwargs)
    return model


def main():
    logger = get_logger('RUN_FASTV')
    args = parse_args()
    
    # Load config from JSON if provided
    if args.config is not None:
        config = load_config_from_json(args.config)
        # Override args with config values
        for key, value in config.items():
            setattr(args, key, value)
    
    # Set model name
    if args.model_name is None:
        args.model_name = osp.basename(args.model_path.rstrip('/'))
        if args.use_fast_v:
            args.model_name += f"_fastv_rank{args.fast_v_attention_rank}"
    
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Datasets: {args.data}")
    logger.info(f"Work directory: {args.work_dir}")
    
    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Build model
    model = build_fastv_model(args)
    
    # Evaluate on each dataset
    for dataset_name in args.data:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating on dataset: {dataset_name}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Build dataset
            dataset = build_dataset(dataset_name)
            if dataset is None:
                logger.error(f'Dataset {dataset_name} is not valid, will be skipped.')
                continue
            
            # Create output directory for this evaluation
            eval_id = f"{args.model_name}_{dataset_name}"
            pred_root = osp.join(args.work_dir, eval_id)
            os.makedirs(pred_root, exist_ok=True)
            
            # Run inference
            if args.mode in ['all', 'infer']:
                logger.info(f"Running inference on {dataset_name}...")
                model = infer_data_job(
                    model,
                    work_dir=pred_root,
                    model_name=args.model_name,
                    dataset=dataset,
                    verbose=args.verbose,
                    api_nproc=1,  # Not using API
                    ignore_failed=False,
                )
            
            # Run evaluation
            if args.mode in ['all', 'eval']:
                logger.info(f"Evaluating results on {dataset_name}...")
                from vlmeval.utils import get_pred_file_format
                pred_format = get_pred_file_format()
                result_file = osp.join(pred_root, f'{args.model_name}_{dataset_name}.{pred_format}')
                
                if not osp.exists(result_file):
                    logger.warning(f"Result file {result_file} not found. Skipping evaluation.")
                    continue
                
                eval_results = dataset.evaluate(result_file, nproc=4, verbose=args.verbose)
                
                # Display results
                if eval_results is not None:
                    logger.info(f'\nEvaluation Results for {dataset_name}:')
                    logger.info('='*80)
                    if isinstance(eval_results, dict):
                        logger.info('\n' + json.dumps(eval_results, indent=4))
                    elif isinstance(eval_results, pd.DataFrame):
                        from tabulate import tabulate
                        if len(eval_results) < len(eval_results.columns):
                            eval_results = eval_results.T
                        logger.info('\n' + tabulate(eval_results))
                    logger.info('='*80 + '\n')
                    
                    # Save results to JSON
                    result_json_path = osp.join(pred_root, f'{args.model_name}_{dataset_name}_results.json')
                    if isinstance(eval_results, dict):
                        with open(result_json_path, 'w') as f:
                            json.dump(eval_results, f, indent=4)
                    else:
                        eval_results.to_json(result_json_path, indent=4)
                    logger.info(f"Results saved to {result_json_path}")
                    
        except Exception as e:
            logger.exception(f'Evaluation on {dataset_name} failed: {e}')
            continue
    
    logger.info("\n" + "="*80)
    logger.info("All evaluations completed!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
