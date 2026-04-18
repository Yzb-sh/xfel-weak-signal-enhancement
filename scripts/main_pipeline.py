#!/usr/bin/env python3
"""
DeepPhase-X Main Pipeline.

Main entry script for the XFEL weak signal enhancement and phasing system.
Supports data generation, training, validation, and inference workflows.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local imports
from src.config.config_loader import (
    ConfigLoader,
    SimulationConfig,
    ModelConfig,
    GANConfig,
    TrainingConfig,
)
from src.physics.simulator import XRaySimulator
from src.physics.noise_model import AnalyticNoiseModel
from src.models.unet_physics import PhysicsUNet, LightweightPhysicsUNet
from src.models.noise_gan import NoiseGAN
from src.models.losses import CombinedLoss
from src.reconstruction.hio_er import PhaseRetrieval
from src.reconstruction.support import SupportEstimator
from src.utils.metrics import DiffractionMetrics, PhaseRetrievalMetrics, NoiseMetrics
from src.utils.checkpoint import CheckpointManager
from src.utils.visualization import (
    setup_publication_style,
    create_summary_figure,
    plot_training_history,
    save_figure,
)
from src.data.dataset import DiffractionDataset, save_to_hdf5, load_from_hdf5


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    config: SimulationConfig,
    output_dir: Path,
    n_samples: int = 100,
    exposure_levels: list = None
) -> Path:
    """
    Generate synthetic diffraction data.
    
    Args:
        config: Simulation configuration
        output_dir: Output directory for HDF5 files
        n_samples: Number of samples to generate
        exposure_levels: List of exposure levels for noise
        
    Returns:
        Path to generated HDF5 file
    """
    logger.info(f"Generating {n_samples} synthetic diffraction patterns...")
    
    if exposure_levels is None:
        exposure_levels = [10, 50, 100, 500, 1000]
    
    simulator = XRaySimulator(config)
    noise_model = AnalyticNoiseModel()
    
    clean_patterns = []
    noisy_patterns = []
    densities = []
    
    for i in range(n_samples):
        # Generate random density
        density = simulator.create_synthetic_density(
            size=config.grid_size,
            n_objects=np.random.randint(1, 5),
            object_type='mixed'
        )
        
        # Generate diffraction pattern
        clean = simulator.generate_diffraction(density)
        
        # Add noise at random exposure level
        exposure = np.random.choice(exposure_levels)
        noisy = noise_model.add_poisson_gaussian(
            clean,
            exposure_level=exposure,
            readout_noise=2.0
        )
        
        clean_patterns.append(clean)
        noisy_patterns.append(noisy)
        densities.append(density)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1}/{n_samples} samples")
    
    # Save to HDF5
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_dataset.h5"
    
    data = {
        'clean': np.array(clean_patterns),
        'noisy': np.array(noisy_patterns),
        'density': np.array(densities),
    }
    
    save_to_hdf5(data, str(output_path))
    logger.info(f"Saved dataset to {output_path}")
    
    return output_path


def train_denoiser(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_data_path: Path,
    val_data_path: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    device: str = "cuda"
) -> dict:
    """
    Train the denoising model.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        train_data_path: Path to training HDF5 file
        val_data_path: Optional path to validation HDF5 file
        checkpoint_dir: Directory for saving checkpoints
        device: Device to train on
        
    Returns:
        Training history dictionary
    """
    logger.info("Starting denoiser training...")
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Load data
    train_data = load_from_hdf5(str(train_data_path))
    train_dataset = DiffractionDataset(
        noisy_patterns=train_data['noisy'],
        clean_patterns=train_data['clean']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    model = PhysicsUNet(model_config).to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(
        use_pnll=training_config.loss_type == "pnll",
        use_freq_weight=training_config.freq_weight
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    
    # Checkpoint manager
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_manager = CheckpointManager(str(checkpoint_dir), max_checkpoints=5)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(training_config.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Prepare dual input (intensity and amplitude)
            intensity = noisy
            amplitude = torch.sqrt(torch.abs(noisy) + 1e-8)
            
            output = model(intensity, amplitude)
            loss = criterion(output, clean)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        if val_data_path:
            val_loss = validate_model(model, val_data_path, criterion, device)
            history['val_loss'].append(val_loss)
            logger.info(f"Epoch {epoch + 1}/{training_config.epochs} - "
                       f"Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        else:
            logger.info(f"Epoch {epoch + 1}/{training_config.epochs} - "
                       f"Train Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            ckpt_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss
            )
    
    logger.info("Training completed!")
    return history


def validate_model(
    model: torch.nn.Module,
    data_path: Path,
    criterion: torch.nn.Module,
    device: str
) -> float:
    """Validate model on a dataset."""
    model.eval()
    
    data = load_from_hdf5(str(data_path))
    dataset = DiffractionDataset(
        noisy_patterns=data['noisy'],
        clean_patterns=data['clean']
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    total_loss = 0.0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            intensity = noisy
            amplitude = torch.sqrt(torch.abs(noisy) + 1e-8)
            
            output = model(intensity, amplitude)
            loss = criterion(output, clean)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def closed_loop_validation(
    model: torch.nn.Module,
    test_data_path: Path,
    output_dir: Path,
    device: str = "cuda"
) -> dict:
    """
    Perform closed-loop validation.
    
    Clean → Add Noise → Denoise → Phase Retrieval → Compare
    
    Args:
        model: Trained denoising model
        test_data_path: Path to test data
        output_dir: Output directory for results
        device: Device for inference
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Starting closed-loop validation...")
    
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    data = load_from_hdf5(str(test_data_path))
    clean_patterns = data['clean']
    noisy_patterns = data['noisy']
    densities = data.get('density', None)
    
    # Phase retrieval setup
    phase_retrieval = PhaseRetrieval(beta=0.9, max_iter=500)
    support_estimator = SupportEstimator()
    
    # Metrics
    metrics = {
        'r_factor_noisy': [],
        'r_factor_denoised': [],
        'psnr_noisy': [],
        'psnr_denoised': [],
        'ssim_noisy': [],
        'ssim_denoised': [],
    }
    
    n_samples = min(len(clean_patterns), 10)  # Limit for validation
    
    for i in range(n_samples):
        clean = clean_patterns[i]
        noisy = noisy_patterns[i]
        
        # Denoise
        with torch.no_grad():
            noisy_tensor = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)
            amplitude_tensor = torch.sqrt(torch.abs(noisy_tensor) + 1e-8)
            denoised_tensor = model(noisy_tensor, amplitude_tensor)
            denoised = denoised_tensor.squeeze().cpu().numpy()
        
        # Compute metrics
        metrics['r_factor_noisy'].append(DiffractionMetrics.r_factor(noisy, clean))
        metrics['r_factor_denoised'].append(DiffractionMetrics.r_factor(denoised, clean))
        metrics['psnr_noisy'].append(DiffractionMetrics.psnr(noisy, clean))
        metrics['psnr_denoised'].append(DiffractionMetrics.psnr(denoised, clean))
        metrics['ssim_noisy'].append(DiffractionMetrics.ssim(noisy, clean))
        metrics['ssim_denoised'].append(DiffractionMetrics.ssim(denoised, clean))
        
        # Phase retrieval comparison (first sample only for speed)
        if i == 0:
            magnitude_clean = np.sqrt(np.maximum(clean, 0))
            magnitude_denoised = np.sqrt(np.maximum(denoised, 0))
            
            support = support_estimator.from_autocorrelation(clean, threshold=0.1)
            
            recon_clean, _ = phase_retrieval.hybrid(magnitude_clean, support)
            recon_denoised, _ = phase_retrieval.hybrid(magnitude_denoised, support)
            
            # Save visualization
            original_density = densities[i] if densities is not None else None
            fig = create_summary_figure(
                clean, noisy, denoised,
                reconstructed_density=recon_denoised,
                original_density=original_density,
                metrics={
                    'R-factor (noisy)': metrics['r_factor_noisy'][-1],
                    'R-factor (denoised)': metrics['r_factor_denoised'][-1],
                    'PSNR (denoised)': metrics['psnr_denoised'][-1],
                },
                save_path=str(output_dir / f"sample_{i}_summary.png")
            )
            plt.close(fig)
    
    # Compute average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    logger.info("Closed-loop validation results:")
    for k, v in avg_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    return avg_metrics


def sim_to_real_validation(
    model: torch.nn.Module,
    gan_model: Optional[NoiseGAN],
    test_data_path: Path,
    output_dir: Path,
    device: str = "cuda"
) -> dict:
    """
    Perform Sim-to-Real validation.
    
    Train on simulated data, test on GAN-generated or real data.
    
    Args:
        model: Trained denoising model
        gan_model: Optional trained NoiseGAN for generating realistic noise
        test_data_path: Path to test data
        output_dir: Output directory
        device: Device for inference
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Starting Sim-to-Real validation...")
    
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    data = load_from_hdf5(str(test_data_path))
    clean_patterns = data['clean']
    
    metrics = {
        'r_factor_analytic': [],
        'r_factor_gan': [],
        'psnr_analytic': [],
        'psnr_gan': [],
    }
    
    noise_model = AnalyticNoiseModel()
    
    n_samples = min(len(clean_patterns), 10)
    
    for i in range(n_samples):
        clean = clean_patterns[i]
        
        # Add analytic noise
        noisy_analytic = noise_model.add_poisson_gaussian(clean, exposure_level=100)
        
        # Denoise analytic noise
        with torch.no_grad():
            noisy_tensor = torch.from_numpy(noisy_analytic).float().unsqueeze(0).unsqueeze(0).to(device)
            amplitude_tensor = torch.sqrt(torch.abs(noisy_tensor) + 1e-8)
            denoised_analytic = model(noisy_tensor, amplitude_tensor).squeeze().cpu().numpy()
        
        metrics['r_factor_analytic'].append(DiffractionMetrics.r_factor(denoised_analytic, clean))
        metrics['psnr_analytic'].append(DiffractionMetrics.psnr(denoised_analytic, clean))
        
        # Add GAN noise if available
        if gan_model is not None:
            gan_model.eval()
            with torch.no_grad():
                clean_tensor = torch.from_numpy(clean).float().unsqueeze(0).unsqueeze(0).to(device)
                noisy_gan = gan_model.generator(clean_tensor).squeeze().cpu().numpy()
                
                noisy_tensor = torch.from_numpy(noisy_gan).float().unsqueeze(0).unsqueeze(0).to(device)
                amplitude_tensor = torch.sqrt(torch.abs(noisy_tensor) + 1e-8)
                denoised_gan = model(noisy_tensor, amplitude_tensor).squeeze().cpu().numpy()
            
            metrics['r_factor_gan'].append(DiffractionMetrics.r_factor(denoised_gan, clean))
            metrics['psnr_gan'].append(DiffractionMetrics.psnr(denoised_gan, clean))
    
    # Compute averages
    avg_metrics = {}
    for k, v in metrics.items():
        if v:
            avg_metrics[k] = np.mean(v)
    
    logger.info("Sim-to-Real validation results:")
    for k, v in avg_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    return avg_metrics


def curriculum_training(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    gan_config: GANConfig,
    train_data_path: Path,
    checkpoint_dir: Path,
    device: str = "cuda"
) -> dict:
    """
    Curriculum learning training strategy.
    
    Phase 1: Pre-train on analytic noise
    Phase 2: Fine-tune on GAN noise
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        gan_config: GAN configuration
        train_data_path: Path to training data
        checkpoint_dir: Checkpoint directory
        device: Training device
        
    Returns:
        Combined training history
    """
    logger.info("Starting curriculum learning...")
    
    # Phase 1: Analytic noise pre-training
    logger.info("Phase 1: Pre-training on analytic noise...")
    phase1_epochs = training_config.epochs // 2
    
    phase1_config = TrainingConfig(
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        epochs=phase1_epochs,
        loss_type=training_config.loss_type,
        freq_weight=training_config.freq_weight,
    )
    
    history_phase1 = train_denoiser(
        model_config=model_config,
        training_config=phase1_config,
        train_data_path=train_data_path,
        checkpoint_dir=checkpoint_dir / "phase1",
        device=device
    )
    
    # Phase 2: GAN noise fine-tuning
    logger.info("Phase 2: Fine-tuning on GAN noise...")
    phase2_epochs = training_config.epochs - phase1_epochs
    
    phase2_config = TrainingConfig(
        batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate * 0.1,  # Lower LR for fine-tuning
        epochs=phase2_epochs,
        loss_type=training_config.loss_type,
        freq_weight=training_config.freq_weight,
    )
    
    # TODO: Generate GAN noise data and fine-tune
    # For now, continue with analytic noise
    history_phase2 = train_denoiser(
        model_config=model_config,
        training_config=phase2_config,
        train_data_path=train_data_path,
        checkpoint_dir=checkpoint_dir / "phase2",
        device=device
    )
    
    # Combine histories
    history = {
        'train_loss': history_phase1['train_loss'] + history_phase2['train_loss'],
        'phase1_epochs': phase1_epochs,
        'phase2_epochs': phase2_epochs,
    }
    
    logger.info("Curriculum learning completed!")
    return history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DeepPhase-X: XFEL Weak Signal Enhancement & Phasing"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate data command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--output', type=str, default='data/simulated_h5',
                           help='Output directory')
    gen_parser.add_argument('--n-samples', type=int, default=100,
                           help='Number of samples to generate')
    gen_parser.add_argument('--config', type=str, default='configs/simulation_config.yaml',
                           help='Simulation config file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train denoising model')
    train_parser.add_argument('--data', type=str, required=True,
                             help='Path to training data HDF5')
    train_parser.add_argument('--val-data', type=str, default=None,
                             help='Path to validation data HDF5')
    train_parser.add_argument('--checkpoint-dir', type=str, default='experiments/checkpoints',
                             help='Checkpoint directory')
    train_parser.add_argument('--config-dir', type=str, default='configs',
                             help='Configuration directory')
    train_parser.add_argument('--device', type=str, default='cuda',
                             help='Device (cuda/cpu)')
    train_parser.add_argument('--curriculum', action='store_true',
                             help='Use curriculum learning')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Validate trained model')
    val_parser.add_argument('--checkpoint', type=str, required=True,
                           help='Path to model checkpoint')
    val_parser.add_argument('--data', type=str, required=True,
                           help='Path to test data HDF5')
    val_parser.add_argument('--output', type=str, default='experiments/validation',
                           help='Output directory')
    val_parser.add_argument('--device', type=str, default='cuda',
                           help='Device (cuda/cpu)')
    val_parser.add_argument('--mode', type=str, default='closed-loop',
                           choices=['closed-loop', 'sim-to-real'],
                           help='Validation mode')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        # Load config
        config = SimulationConfig()
        if Path(args.config).exists():
            config = ConfigLoader.load_simulation_config(args.config)
        
        generate_synthetic_data(
            config=config,
            output_dir=Path(args.output),
            n_samples=args.n_samples
        )
    
    elif args.command == 'train':
        # Load configs
        config_dir = Path(args.config_dir)
        model_config = ConfigLoader.load_model_config(
            str(config_dir / 'model_config.yaml')
        ) if (config_dir / 'model_config.yaml').exists() else ModelConfig()
        
        training_config = ConfigLoader.load_training_config(
            str(config_dir / 'training_config.yaml')
        ) if (config_dir / 'training_config.yaml').exists() else TrainingConfig()
        
        if args.curriculum:
            gan_config = ConfigLoader.load_gan_config(
                str(config_dir / 'gan_config.yaml')
            ) if (config_dir / 'gan_config.yaml').exists() else GANConfig()
            
            curriculum_training(
                model_config=model_config,
                training_config=training_config,
                gan_config=gan_config,
                train_data_path=Path(args.data),
                checkpoint_dir=Path(args.checkpoint_dir),
                device=args.device
            )
        else:
            train_denoiser(
                model_config=model_config,
                training_config=training_config,
                train_data_path=Path(args.data),
                val_data_path=Path(args.val_data) if args.val_data else None,
                checkpoint_dir=Path(args.checkpoint_dir),
                device=args.device
            )
    
    elif args.command == 'validate':
        # Load model
        from src.utils.checkpoint import load_checkpoint
        
        config_dir = Path(args.checkpoint).parent.parent / 'configs'
        model_config = ConfigLoader.load_model_config(
            str(config_dir / 'model_config.yaml')
        ) if (config_dir / 'model_config.yaml').exists() else ModelConfig()
        
        model = PhysicsUNet(model_config)
        load_checkpoint(model, args.checkpoint)
        model = model.to(args.device)
        
        if args.mode == 'closed-loop':
            closed_loop_validation(
                model=model,
                test_data_path=Path(args.data),
                output_dir=Path(args.output),
                device=args.device
            )
        else:
            sim_to_real_validation(
                model=model,
                gan_model=None,
                test_data_path=Path(args.data),
                output_dir=Path(args.output),
                device=args.device
            )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    # Import matplotlib here to avoid issues when not using visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    main()
