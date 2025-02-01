import logging
import sys
import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import time

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
    """Training loop with zero-trust integration and performance analysis"""
    num_epochs = config['num_epochs']
    
    # Initialize performance analyzer and ensure plots directory exists
    plots_dir = Path(config['plots_dir'])
    plots_dir.mkdir(parents=True, exist_ok=True)
    analyzer = PerformanceAnalyzer(save_dir=plots_dir)
    logger.info(f"Performance analyzer initialized. Plots will be saved to: {plots_dir.absolute()}")
    
    for epoch in range(num_epochs):
        try:
            # Start epoch timer
            epoch_start_time = time.time()
            
            # Train for one epoch
            logger.info(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
            
            # Train with gradient collection
            model.train()
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(trainer.device), target.to(trainer.device)
                trainer.optimizer.zero_grad()
                output = model(data)
                loss = trainer.criterion(output, target)
                loss.backward()
                
                # Collect gradients before optimizer step
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if name not in analyzer.gradient_norms:
                                analyzer.gradient_norms[name] = []
                            analyzer.gradient_norms[name].append(param.grad.norm().item())
                
                trainer.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Evaluate
            val_accuracy, tpr, fpr, predictions = trainer.evaluate(val_loader)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update performance metrics
            metrics = {
                'train_loss': avg_loss,
                'val_loss': trainer.evaluate_loss(val_loader),
                'train_acc': trainer.get_last_train_accuracy(),
                'val_acc': val_accuracy,
                'epoch_time': epoch_time
            }
            analyzer.update_metrics(metrics, epoch)
            
            # Generate plots every epoch
            logger.info("Generating plots...")
            analyzer.plot_training_curves()
            analyzer.plot_gradient_norms()
            analyzer.plot_resource_usage()
            analyzer.plot_weight_distribution(model, epoch + 1)
            logger.info(f"Plots updated in: {plots_dir.absolute()}")
            
            # Update zero-trust thresholds
            stats_dict = {'TPR': tpr, 'FPR': fpr}
            zt_controller.dynamic_update_thresholds(stats_dict)
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} Results:\n"
                f"  Loss: {avg_loss:.4f}\n"
                f"  Validation Accuracy: {val_accuracy:.4f}\n"
                f"  TPR: {tpr:.4f}\n"
                f"  FPR: {fpr:.4f}\n"
                f"  Time: {epoch_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Error in epoch {epoch + 1}: {str(e)}", exc_info=True)
            continue
    
    # Final plots and report
    logger.info("\nTraining completed. Generating final plots and report...")
    try:
        # Force plot generation one last time
        analyzer.plot_training_curves()
        analyzer.plot_gradient_norms()
        analyzer.plot_resource_usage()
        analyzer.plot_weight_distribution(model, num_epochs)
        
        performance_summary = analyzer.generate_performance_report()
        
        # List all generated plots
        logger.info("\nGenerated plot files:")
        for plot_file in plots_dir.glob("*.png"):
            logger.info(f"- {plot_file.absolute()}")
        
        return performance_summary
    except Exception as e:
        logger.error(f"Error generating final plots and report: {str(e)}")
        return None

def main():
    try:
        # Initialize logger correctly
        logger = LoggerConfig.setup_logger(
            "main",
            log_file='logs/casnet.log',
            level=logging.INFO
        )
        logger.info("Starting CasNet application")

        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

        # Load configuration
        config = load_config('config.yaml')
        logger.info("Configuration loaded successfully")
        
        # Initialize components
        preprocessor = DataPreprocessor(config_path='config.yaml')
        checkpoint_manager = CheckpointManager(save_dir='checkpoints')
        visualizer = TrainingVisualizer(save_dir='plots')

        # Prepare data
        X_train, y_train, X_test, y_test = preprocessor.prepare_data(
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        logger.info(f"Data prepared - Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Save scaler for future use
        preprocessor.save_scaler('models/scaler.pkl')
        logger.info("Scaler saved successfully")

        # Create datasets and data loaders
        train_dataset = CasNet2024Dataset(X_train, y_train)
        test_dataset = CasNet2024Dataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        logger.info("Data loaders created successfully")

        # Initialize model with your YAML structure
        model = CVQNNClassifier(
            in_features=config['in_features'],
            n_layers=config['n_layers'],
            layer_widths=config['layer_widths'],
            out_classes=config['num_classes'],
            cutoff=config['cutoff_dim']
        )
        logger.info("Model initialized successfully")
        
        # Initialize trainer with your YAML structure
        trainer = ModelTrainer(
            model=model,
            config=config,
            checkpoint_manager=checkpoint_manager,
            visualizer=visualizer
        )
        logger.info("Trainer initialized successfully")
        
        # Initialize zero-trust controller with default values since they're not in your YAML
        zt_controller = ZeroTrustController(
            gamma_init=0.5,  # default value
            tau_init=0.5     # default value
        )
        logger.info("Zero-Trust controller initialized successfully")
        
        # Train the model
        train(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            trainer=trainer,
            zt_controller=zt_controller,
            config=config,
            logger=logger
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        raise

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