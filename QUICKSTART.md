# Quick Start Guide

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

## Training

### Basic Training

Train with default configuration (uses synthetic data):
```bash
python scripts/train.py
```

### Custom Configuration

Train with specific settings:
```bash
python scripts/train.py --config configs/default.yaml --epochs 50 --device cuda
```

### Monitor Training

Training progress is logged to:
- Console output (real-time)
- `logs/training.log` (file)
- `mlruns/` (MLflow tracking)
- Checkpoints saved to `checkpoints/`

## Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

Save results to custom location:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --output results/my_results.json
```

## Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=urban_occlusion_aware_segmentation --cov-report=html
```

## Using Cityscapes Dataset

1. Download Cityscapes from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)
2. Extract to `data/cityscapes/`
3. Update `configs/default.yaml`:
```yaml
data:
  root_dir: "./data/cityscapes"
  use_synthetic: false
```

## Exploring the Code

Open the Jupyter notebook:
```bash
jupyter notebook notebooks/exploration.ipynb
```

## Project Structure

```
urban-occlusion-aware-segmentation/
├── src/urban_occlusion_aware_segmentation/  # Main package
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model implementations (SegFormer, DeepLabV3+, Ensemble)
│   ├── training/       # Training loop and custom loss functions
│   ├── evaluation/     # Metrics (mIoU, boundary F1, occlusion recall)
│   └── utils/          # Configuration and utilities
├── scripts/
│   ├── train.py        # Training script - START HERE
│   └── evaluate.py     # Evaluation script
├── configs/
│   └── default.yaml    # Configuration file
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks
└── checkpoints/        # Saved models (created during training)
```

## Key Features to Explore

1. **Custom Occlusion-Weighted Loss** (`src/urban_occlusion_aware_segmentation/training/trainer.py`)
   - Automatically emphasizes boundary regions
   - Configurable boundary width and weight multiplier

2. **Uncertainty Quantification** (`src/urban_occlusion_aware_segmentation/models/model.py`)
   - Ensemble disagreement as uncertainty measure
   - Useful for identifying high-risk predictions

3. **Comprehensive Metrics** (`src/urban_occlusion_aware_segmentation/evaluation/metrics.py`)
   - Standard mIoU
   - Boundary-specific F1 score
   - Occlusion recall for safety-critical classes

## Configuration Tips

Edit `configs/default.yaml` to customize:

- **Model architecture**: Change `model.type` to "segformer", "deeplabv3plus", or "ensemble"
- **Batch size**: Adjust `training.batch_size` based on GPU memory
- **Learning rate**: Tune `training.learning_rate` (default: 0.0001)
- **Image size**: Modify `data.image_size` for speed/accuracy tradeoff
- **Loss weights**: Balance primary and auxiliary losses via `loss.weights`

## Troubleshooting

**CUDA out of memory:**
- Reduce `training.batch_size`
- Decrease `data.image_size`
- Disable mixed precision: `training.amp: false`

**Slow training:**
- Enable mixed precision: `training.amp: true`
- Increase `system.num_workers`
- Use smaller image size

**MLflow errors:**
- MLflow is optional - training continues even if MLflow fails
- Check `logging.mlflow.tracking_uri` is writable

## Next Steps

1. Run training: `python scripts/train.py`
2. Monitor in `logs/training.log`
3. Evaluate results: `python scripts/evaluate.py --checkpoint checkpoints/best_model.pth`
4. Tune hyperparameters in `configs/default.yaml`
5. Explore predictions in `notebooks/exploration.ipynb`
