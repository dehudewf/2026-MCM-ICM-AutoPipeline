#!/bin/bash
################################################################################
# MCM 2026 Problem A - Quick Run Script
################################################################################
# This script provides convenient shortcuts for running the battery modeling
# pipeline with different configurations.
#
# Usage:
#   ./quick_run.sh              # Standard run
#   ./quick_run.sh verbose      # Verbose logging
#   ./quick_run.sh clean        # Clean output directory first
#   ./quick_run.sh test         # Test mode (dry-run)
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  MCM 2026 Problem A - Battery Modeling Pipeline${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Parse command
MODE="${1:-standard}"

case "$MODE" in
    "clean")
        echo -e "${YELLOW}[CLEAN MODE]${NC} Removing old outputs..."
        rm -rf output/*.png output/*.json output/*.csv output/*.md
        echo -e "${GREEN}✓${NC} Output directory cleaned"
        echo ""
        echo -e "${BLUE}Running pipeline...${NC}"
        python3 run_pipeline.py
        ;;
    
    "verbose")
        echo -e "${YELLOW}[VERBOSE MODE]${NC} Running with detailed logging..."
        python3 run_pipeline.py --verbose
        ;;
    
    "test")
        echo -e "${YELLOW}[TEST MODE]${NC} Checking imports and syntax..."
        python3 -c "
import sys
sys.path.insert(0, '.')
from src.pipeline import MCMBatteryPipeline
from src.config import logger
logger.info('✓ All imports successful')
print('${GREEN}✓${NC} Test passed - pipeline is ready to run')
"
        ;;
    
    "standard"|*)
        echo -e "${YELLOW}[STANDARD MODE]${NC} Running pipeline..."
        python3 run_pipeline.py
        ;;
esac

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================================${NC}"
    echo -e "${GREEN}  ✓ Pipeline Execution Complete${NC}"
    echo -e "${GREEN}=================================================${NC}"
    echo ""
    echo -e "${BLUE}Generated Files:${NC}"
    ls -lh output/ | tail -n +2
    echo ""
    echo -e "${BLUE}Quick View:${NC}"
    echo -e "  Summary Report: ${GREEN}output/mcm_2026_summary_report.md${NC}"
    echo -e "  Full Results:   ${GREEN}output/mcm_2026_results.json${NC}"
    echo -e "  TTE Predictions: ${GREEN}output/tte_predictions.csv${NC}"
    echo ""
    echo -e "${YELLOW}Tip:${NC} View summary with: ${BLUE}cat output/mcm_2026_summary_report.md${NC}"
else
    echo ""
    echo -e "${RED}=================================================${NC}"
    echo -e "${RED}  ✗ Pipeline Execution Failed${NC}"
    echo -e "${RED}=================================================${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "  1. Check data directory exists: ${BLUE}battery_data/${NC}"
    echo -e "  2. Verify Python dependencies are installed"
    echo -e "  3. Run in test mode: ${BLUE}./quick_run.sh test${NC}"
    exit 1
fi
