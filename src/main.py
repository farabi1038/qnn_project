import logging
import sys
import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from data import DataPreprocessor, CasNet2024Dataset
from models import ModelTrainer, CheckpointManager, CVQNNClassifier
from utils import TrainingVisualizer
from zero_trust import ZeroTrustController
from logger_config import LoggerConfig

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading config: {str(e)}")

def setup_directories(config: dict) -> None:
    """Create necessary directories for outputs."""
    directories = [
        config['checkpoint_dir'],
        config['plots_dir'],
        'logs',
        'models'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    """Main execution function"""
    # Create a basic logger first for error handling
    logging.basicConfig(level=logging.INFO)
    basic_logger = logging.getLogger(__name__)
    
    try:
        # Create necessary directories first
        Path('logs').mkdir(parents=True, exist_ok=True)
        
        # Now setup the full logger
        logger = LoggerConfig.setup_logger(
            "main",
            log_file='logs/casnet.log',
            level=logging.INFO
        )
        logger.info("Starting CasNet application")

        # Load configuration
        config = load_config('config.yaml')
        logger.info("Configuration loaded successfully")

        # Setup all other directories
        setup_directories(config)
        logger.info("Directory structure created")

        # Initialize data preprocessor
        preprocessor = DataPreprocessor(config_path='config.yaml')
        logger.info("Data preprocessor initialized")

        # Prepare data
        X_train, _, X_test, _ = preprocessor.prepare_data(
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        logger.info(f"Data prepared - Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Save scaler for future use
        preprocessor.save_scaler('models/scaler.pkl')
        logger.info("Scaler saved successfully")

        # Create datasets
        train_dataset = CasNet2024Dataset(X_train)
        test_dataset = CasNet2024Dataset(X_test)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 2),
            pin_memory=True if torch.cuda.is_available() else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 2),
            pin_memory=True if torch.cuda.is_available() else False
        )
        logger.info("Data loaders created successfully")

        # Initialize model
        model = CVQNNClassifier(
            in_features=X_train.shape[1],  # Use actual number of features
            n_layers=config['n_layers'],
            layer_widths=config['layer_widths'],
            out_classes=config['num_classes'],
            cutoff=config['cutoff_dim']
        )
        logger.info("Model initialized")

        # Initialize checkpoint manager and visualizer
        checkpoint_manager = CheckpointManager(save_dir=config['checkpoint_dir'])
        visualizer = TrainingVisualizer(save_dir=config['plots_dir'])
        logger.info("Checkpoint manager and visualizer initialized")

        # Initialize trainer
        trainer = ModelTrainer(
            model=model,
            config=config,
            checkpoint_manager=checkpoint_manager,
            visualizer=visualizer
        )
        logger.info("Trainer initialized")

        # Initialize zero-trust controller
        zt_controller = ZeroTrustController(
            gamma_init=config.get('zero_trust', {}).get('gamma_init', 0.4),
            tau_init=config.get('zero_trust', {}).get('tau_init', 0.5)
        )
        logger.info("Zero-trust controller initialized")

        # Resume training if checkpoint exists
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            checkpoint_manager.load_checkpoint(
                latest_checkpoint,
                model=model,
                optimizer=trainer.optimizer
            )
            logger.info(f"Resumed training from checkpoint: {latest_checkpoint}")

        # Training loop
        logger.info("Starting training...")
        for epoch in range(trainer.current_epoch, config['num_epochs']):
            try:
                # Training step
                avg_loss = trainer.train_epoch(train_loader)
                logger.info(f"Training loss: {avg_loss:.4f}")
                # Validation step
                val_acc, tpr, fpr, _ = trainer.evaluate(test_loader)
                logger.info(f"Validation accuracy: {val_acc:.4f}")
                # Update zero-trust thresholds
                stats_dict = {'TPR': tpr, 'FPR': fpr}
                zt_controller.dynamic_update_thresholds(stats_dict)
                logger.info(f"Updated zero-trust thresholds: γ_q={zt_controller.gamma_q:.2f}, τ={zt_controller.tau:.2f}")
                # Update visualization history
                visualizer.update_history(
                    epoch=epoch,
                    loss=avg_loss,
                    accuracy=val_acc,
                    learning_rate=trainer.optimizer.param_groups[0]['lr'],
                    tpr=tpr,
                    fpr=fpr
                )
                
                # Save checkpoint
                if (epoch + 1) % config['checkpoint_frequency'] == 0:
                    metrics = {
                        'loss': avg_loss,
                        'accuracy': val_acc,
                        'tpr': tpr,
                        'fpr': fpr
                    }
                    checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=trainer.optimizer,
                        epoch=epoch,
                        metrics=metrics,
                        is_best=(val_acc > trainer.best_accuracy)
                    )
                
                # Log progress
                logger.info(
                    f"Epoch {epoch+1}/{config['num_epochs']} - "
                    f"Loss: {avg_loss:.4f}, Acc: {val_acc:.4f}, "
                    f"TPR: {tpr:.4f}, FPR: {fpr:.4f}, "
                    f"γ_q: {zt_controller.gamma_q:.2f}, "
                    f"τ: {zt_controller.tau:.2f}"
                )
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch+1}: {str(e)}")
                continue

        # Final evaluation
        logger.info("Training completed. Starting final evaluation...")
        perform_final_evaluation(trainer, test_loader, zt_controller)

        # Save final visualizations
        visualizer.plot_metrics(save=True)
        visualizer.save_metrics_csv()
        logger.info("Final visualizations saved")

        logger.info("CasNet application completed successfully")

    except Exception as e:
        basic_logger.error(f"Fatal error in main execution: {str(e)}")
        sys.exit(1)

def perform_final_evaluation(
    trainer: ModelTrainer,
    test_loader: DataLoader,
    zt_controller: ZeroTrustController
) -> None:
    """Perform final model evaluation with zero-trust decisions."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("\n=== FINAL EVALUATION AND ZERO-TRUST DECISIONS ===")
        val_acc, tpr, fpr, predictions = trainer.evaluate(test_loader)
        
        logger.info(f"Final Metrics:")
        logger.info(f"  Accuracy: {val_acc:.4f}")
        logger.info(f"  TPR: {tpr:.4f}")
        logger.info(f"  FPR: {fpr:.4f}")
        logger.info(f"  Final γ_q: {zt_controller.gamma_q:.4f}")
        logger.info(f"  Final τ: {zt_controller.tau:.4f}")
        
        # Detailed zero-trust analysis
        sample_count = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(trainer.device)
                logits = trainer.model(batch_x)
                probs = torch.softmax(logits, dim=1)
                
                for i in range(len(batch_x)):
                    # Simulate user/device context
                    user_ctx = float(torch.rand(1))
                    dev_ctx = float(torch.rand(1))
                    mal_prob = float(probs[i, 2])  # probability of malicious class
                    
                    # Compute risk and decisions
                    risk = zt_controller.compute_risk_score(user_ctx, dev_ctx, mal_prob)
                    access_decision = "GRANTED" if risk < zt_controller.tau else "DENIED"
                    seg_decision = zt_controller.micro_segmentation_policy(
                        f"Segment_{sample_count}",
                        mal_prob
                    )
                    
                    logger.info(
                        f"Flow #{sample_count}:\n"
                        f"  Malicious Prob: {mal_prob:.3f}\n"
                        f"  Risk Score: {risk:.3f}\n"
                        f"  Access: {access_decision}\n"
                        f"  {seg_decision}"
                    )
                    
                    sample_count += 1
                    if sample_count >= 20:  # Limit detailed analysis to 20 samples
                        break
                if sample_count >= 20:
                    break
                    
    except Exception as e:
        logger.error(f"Error in final evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()