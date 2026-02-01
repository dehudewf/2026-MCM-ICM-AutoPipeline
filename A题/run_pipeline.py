#!/usr/bin/env python3
"""
================================================================================
MCM 2026 Problem A: Pipeline Runner
================================================================================

Unified runner script for battery modeling pipeline.

Usage:
    # Original pipeline only
    python run_pipeline.py
    
    # Enhanced pipeline (original + O-Award enhancements)
    python run_pipeline.py --enhanced
    
    # Enhanced with more MC samples
    python run_pipeline.py --enhanced --mc-samples 500
    
    # Enhanced components only (skip original)
    python run_pipeline.py --enhanced-only

Author: MCM Team 2026
================================================================================
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main entry point with enhanced pipeline support."""
    parser = argparse.ArgumentParser(
        description='MCM 2026 Problem A: Battery Modeling Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Original pipeline
  python run_pipeline.py --enhanced         # Original + O-Award enhancements
  python run_pipeline.py --enhanced-only    # O-Award enhancements only
  python run_pipeline.py --enhanced --mc-samples 500  # More MC samples

Enhancements Include:
  - Enhanced 5-state ODE with electrochemical-thermal-aging coupling
  - OU→TTE Monte Carlo uncertainty propagation
  - Sobol power component sensitivity analysis
  - O-Award publication-quality composite figures
        """
    )
    
    parser.add_argument(
        '--enhanced', action='store_true',
        help='Run enhanced pipeline (original + O-Award enhancements)'
    )
    parser.add_argument(
        '--enhanced-only', action='store_true',
        help='Run only enhanced components (skip original pipeline)'
    )
    parser.add_argument(
        '--mc-samples', type=int, default=100,
        help='Monte Carlo samples for OU→TTE propagation (default: 100)'
    )
    parser.add_argument(
        '--data-dir', type=str, default='A题/battery_data',
        help='Path to battery data directory'
    )
    parser.add_argument(
        '--output-dir', type=str, default='A题/output',
        help='Path to output directory'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Decide which pipeline to run
    if args.enhanced or args.enhanced_only:
        # Run enhanced pipeline
        from src.pipeline_enhanced import EnhancedMCMBatteryPipeline
        import logging
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        config = {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'mc_samples': args.mc_samples
        }
        
        pipeline = EnhancedMCMBatteryPipeline(config)
        
        if args.enhanced_only:
            # Run only enhancements
            print("\n" + "=" * 60)
            print("  Running O-Award Enhancements Only")
            print("=" * 60)
            task1 = pipeline.run_task1_enhanced_physics()
            task23 = pipeline.run_task23_ou_uncertainty()
            task3 = pipeline.run_task3_power_sensitivity()
            pipeline.generate_composite_figure(task1, task23, task3)
        else:
            # Run full enhanced pipeline
            results = pipeline.run_full_pipeline_enhanced()
        
        return 0
    else:
        # Run original pipeline
        from src.pipeline import main as original_main
        return original_main()


if __name__ == '__main__':
    sys.exit(main())
