import logging
import sys
import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import time

# Custom modules
from data import DataPreprocessor, CasNet2024Dataset
from models import ModelTrainer, CheckpointManager, CVQNNClassifier
from utils import TrainingVisualizer
from zero_trust import ZeroTrustController
from logger_config import LoggerConfig
# Note: The PerformanceAnalyzer class is defined later in this file
from utils.performance_analysis import PerformanceAnalyzer
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

def train(model, train_loader, val_loader, trainer, zt_controller, config, logger):
    """
    Training loop with zero-trust integration, checkpointing, and performance analysis.
    The analyzer is used to update plots after every epoch and finally generate a comprehensive report.
    """
    num_epochs = config['num_epochs']
    plots_dir = Path(config['plots_dir'])
    plots_dir.mkdir(parents=True, exist_ok=True)
    analyzer = PerformanceAnalyzer(save_dir=plots_dir)
    logger.info(f"Performance analyzer initialized. Plots will be saved to: {plots_dir.absolute()}")

    for epoch in range(num_epochs):
        try:
            epoch_start_time = time.time()
            logger.info(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
            model.train()
            total_loss = 0.0

            # Process each batch
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(trainer.device), target.to(trainer.device)
                trainer.optimizer.zero_grad()
                output = model(data)
                loss = trainer.criterion(output, target)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if param.grad is not None:
                            print(f"{name}: grad norm = {param.grad.norm().item()}")
                        else:
                            print(f"{name} has no grad")
                # Compute and update gradient norms (and plot them)
                analyzer.compute_and_plot_gradient_norms(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


                trainer.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_accuracy, tpr, fpr, predictions = trainer.evaluate(val_loader)
            epoch_time = time.time() - epoch_start_time

            metrics = {
                'train_loss': avg_loss,
                'val_loss': trainer.evaluate_loss(val_loader),
                'train_acc': trainer.get_last_train_accuracy(),
                'val_acc': val_accuracy,
                'epoch_time': epoch_time
            }
            analyzer.update_metrics(metrics, epoch)

            # Generate epoch plots
            logger.info("Generating plots for current epoch...")
            analyzer.plot_training_curves()
            analyzer.plot_resource_usage()
            analyzer.plot_weight_distribution(model, epoch + 1)
            logger.info(f"Plots updated in: {plots_dir.absolute()}")

            # Update Zero-Trust thresholds based on current stats
            stats_dict = {'TPR': tpr, 'FPR': fpr}
            zt_controller.dynamic_update_thresholds(stats_dict)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} Results:\n"
                f"  Loss: {avg_loss:.4f}\n"
                f"  Validation Accuracy: {val_accuracy:.4f}\n"
                f"  TPR: {tpr:.4f}\n"
                f"  FPR: {fpr:.4f}\n"
                f"  Time: {epoch_time:.2f}s"
            )

            # Save checkpoint (every 'save_every' epochs, default every epoch)
            save_every = config.get('save_every', 1)
            if (epoch + 1) % save_every == 0:
                trainer.checkpoint_manager.save_checkpoint(model, trainer.optimizer, metrics, epoch + 1)
                logger.info(f"Model checkpoint saved for epoch {epoch + 1}")

        except Exception as e:
            logger.error(f"Error in epoch {epoch + 1}: {str(e)}", exc_info=True)
            continue

    # Final plots and report
    logger.info("\nTraining completed. Generating final plots and performance report...")
    try:
        analyzer.plot_training_curves()
        analyzer.plot_resource_usage()
        analyzer.plot_weight_distribution(model, num_epochs)
        performance_summary = analyzer.generate_performance_report(model, num_epochs)
        
        logger.info("Final performance report generated. Generated plot files:")
        for plot_file in plots_dir.glob("*.png"):
            logger.info(f"- {plot_file.absolute()}")
            
        return performance_summary
    except Exception as e:
        logger.error(f"Error generating final plots and report: {str(e)}")
        return None

def main():
    """Main function to set up, train, checkpoint, and evaluate the model."""
    try:
        # Initialize the logger
        logger = LoggerConfig.setup_logger(
            "main",
            log_file='logs/casnet.log',
            level=logging.INFO
        )
        logger.info("Starting CasNet application")

        # Create required directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

        # Load configuration and create directories
        config = load_config('config.yaml')
        logger.info("Configuration loaded successfully")
        setup_directories(config)

        # Initialize components: data preprocessor, checkpoint manager, and visualizer
        preprocessor = DataPreprocessor(config_path='config.yaml')
        checkpoint_manager = CheckpointManager(save_dir=config.get('checkpoint_dir', 'checkpoints'))
        visualizer = TrainingVisualizer(save_dir=config.get('plots_dir', 'plots'))

        # Prepare data
        X_train, y_train, X_test, y_test = preprocessor.prepare_data(
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        logger.info(f"Data prepared - Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Save the scaler (for later use)
        preprocessor.save_scaler('models/scaler.pkl')
        logger.info("Scaler saved successfully")

        # Create datasets and dataloaders
        train_dataset = CasNet2024Dataset(X_train, y_train)
        test_dataset = CasNet2024Dataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=8,
            pin_memory=True 
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        logger.info("Data loaders created successfully")

        # Initialize the model and log its architecture
        model = CVQNNClassifier(
            in_features=config['in_features'],
            n_layers=config['n_layers'],
            layer_widths=config['layer_widths'],
            out_classes=config['num_classes'],
            cutoff=config['cutoff_dim']
        )
        logger.info("Model initialized successfully")
        logger.info(f"Model architecture: {model}")
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total number of parameters: {num_params}")

        # Initialize trainer and Zero-Trust controller
        trainer = ModelTrainer(
            model=model,
            config=config,
            checkpoint_manager=checkpoint_manager,
            visualizer=visualizer
        )
        logger.info("Trainer initialized successfully")
        zt_controller = ZeroTrustController(
            gamma_init=0.5,  # default value
            tau_init=0.5     # default value
        )
        logger.info("Zero-Trust controller initialized successfully")
        
        # Start training (which will also generate intermediate and final plots)
        performance_summary = train(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            trainer=trainer,
            zt_controller=zt_controller,
            config=config,
            logger=logger
        )
        logger.info("Training completed successfully!")
        logger.info("Performance Summary:")
        logger.info(performance_summary)

        # Optionally, run final evaluation (including zero-trust decisions)
        perform_final_evaluation(trainer, test_loader, zt_controller)
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        raise

def perform_final_evaluation(trainer: ModelTrainer, test_loader: DataLoader, zt_controller: ZeroTrustController):
    """Perform final evaluation and demonstrate zero-trust decisions."""
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

        # Demonstrate zero-trust decisions on a few test samples
        sample_count = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(trainer.device)
                logits = trainer.model(batch_x)
                probs = torch.softmax(logits, dim=1)

                for i in range(len(batch_x)):
                    user_ctx = float(torch.rand(1))
                    dev_ctx = float(torch.rand(1))
                    mal_prob = float(probs[i, 2])  # assuming index 2 corresponds to a malicious class
                    risk = zt_controller.compute_risk_score(user_ctx, dev_ctx, mal_prob)
                    access_decision = "GRANTED" if risk < zt_controller.tau else "DENIED"
                    logger.info(f"Flow #{sample_count}: Risk Score: {risk:.3f}, Access: {access_decision}")
                    sample_count += 1
                    if sample_count >= 20:
                        break
    except Exception as e:
        logger.error(f"Error in final evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
