import argparse
import warnings
from collections import OrderedDict
import torch
import flwr as fl
from ultralytics import YOLO
from FedYOLO.config import SERVER_CONFIG, YOLO_CONFIG, SPLITS_CONFIG, HOME
from FedYOLO.test.extract_final_save_from_client import extract_results_path
import os
import psutil

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--cid", type=int, required=True)
parser.add_argument("--data_path", type=str, default="./client_0_assets/dummy_data_0/data.yaml")

NUM_CLIENTS = SERVER_CONFIG['max_num_clients']

def get_memory_usage():
    """Get current memory usage information."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    print(f"[Memory] Process: {memory_info.rss / 1024 / 1024:.1f} MB ({memory_percent:.1f}%)")
    print(f"[Memory] System: {system_memory.used / 1024 / 1024 / 1024:.1f} GB / {system_memory.total / 1024 / 1024 / 1024:.1f} GB ({system_memory.percent:.1f}%)")
    
    return memory_info.rss, memory_percent, system_memory.percent

def train(net, data_path, cid, strategy, checkpoint_dir, resume=False):
    # Monitor memory before training
    print(f"[Client {cid}] Memory usage before training:")
    get_memory_usage()
    
    # Ensure checkpoint_dir exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Try to load from checkpoint if available
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_batch') and f.endswith('.pt')]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
            print(f"[Client {cid}] Resuming from checkpoint: {latest_checkpoint}")
            net = YOLO(latest_checkpoint)
            resume = True  # We found a checkpoint, so we can resume
    
    # Train with YOLO's built-in checkpointing and memory optimizations
    # save_period=1 means save every epoch
    net.train(
        data=data_path, 
        epochs=YOLO_CONFIG['epochs'], 
        workers=YOLO_CONFIG['workers'],  # Use config setting
        seed=cid, 
        batch=YOLO_CONFIG['batch_size'], 
        imgsz=YOLO_CONFIG['image_size'],  # Use config setting
        project=checkpoint_dir, 
        name="",
        save_period=1,  # Save every epoch
        save=True,      # Enable saving
        resume=resume,  # Only resume if we have a checkpoint
        cache=YOLO_CONFIG['cache'],  # Use config setting
        amp=YOLO_CONFIG['amp'],      # Use config setting
        device=YOLO_CONFIG['device'] # Use config setting
    )
    
    # Also save a checkpoint after training completes
    final_checkpoint_path = f"{checkpoint_dir}/final_checkpoint.pt"
    torch.save(net.model.state_dict(), final_checkpoint_path)
    print(f"[Client {cid}] Final checkpoint saved: {final_checkpoint_path}")

# Define get_section_parameters as a standalone function
from typing import Tuple
def get_section_parameters(state_dict: OrderedDict) -> Tuple[dict, dict, dict]:
    """Get parameters for each section of the model."""
    # Backbone parameters (early layers through conv layers)
    # backbone corresponds to:
    # (0): Conv
    # (1): Conv
    # (2): C3k2
    # (3): Conv
    # (4): C3k2
    # (5): Conv
    # (6): C3k2
    # (7): Conv
    # (8): C3k2
    backbone_weights = {
        k: v for k, v in state_dict.items()
        if not k.startswith(tuple(f'model.{i}' for i in range(9, 24)))
    }

    # Neck parameters
    # The neck consists of the following layers (by index in the Sequential container):
    # (9): SPPF
    # (10): C2PSA
    # (11): Upsample
    # (12): Concat
    # (13): C3k2
    # (14): Upsample
    # (15): Concat
    # (16): C3k2
    # (17): Conv
    # (18): Concat
    # (19): C3k2
    # (20): Conv
    # (21): Concat
    # (22): C3k2
    neck_weights = {
        k: v for k, v in state_dict.items()
        if k.startswith(tuple(f'model.{i}' for i in range(9, 23)))
    }

    # Head parameters (detection head)
    head_weights = {
        k: v for k, v in state_dict.items()
        if k.startswith('model.23')
    }

    return backbone_weights, neck_weights, head_weights

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data_path, dataset_name, strategy_name):
        # Force CPU usage to avoid GPU memory issues
        self.device = torch.device("cpu")
        self.cid = cid
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name
        self.checkpoint_dir = f"{HOME}/client_{self.cid}_checkpoints"
        self.resume_training = False  # Track if we should resume
        
        # Try to load from various checkpoint sources
        import os
        
        # Priority order for checkpoint loading:
        # 1. Final checkpoint from previous run
        # 2. Latest epoch checkpoint
        # 3. Latest batch checkpoint
        # 4. Default YOLO weights
        # 5. Fresh model
        
        checkpoint_loaded = False
        
        # Try final checkpoint first
        final_checkpoint = f"{self.checkpoint_dir}/final_checkpoint.pt"
        if os.path.exists(final_checkpoint):
            print(f"[Client {self.cid}] Loading from final checkpoint: {final_checkpoint}")
            self.net = YOLO()
            self.net.model.load_state_dict(torch.load(final_checkpoint, map_location='cpu'))
            checkpoint_loaded = True
            self.resume_training = True
        
        # Try latest epoch checkpoint
        elif os.path.exists(f"{self.checkpoint_dir}/train"):
            train_dir = f"{self.checkpoint_dir}/train"
            epoch_files = [f for f in os.listdir(train_dir) if f.startswith('epoch') and f.endswith('.pt')]
            if epoch_files:
                latest_epoch = sorted(epoch_files)[-1]
                epoch_path = os.path.join(train_dir, latest_epoch)
                print(f"[Client {self.cid}] Loading from epoch checkpoint: {epoch_path}")
                self.net = YOLO(epoch_path)
                checkpoint_loaded = True
                self.resume_training = True
        
        # Try latest batch checkpoint
        elif os.path.exists(self.checkpoint_dir):
            batch_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_batch') and f.endswith('.pt')]
            if batch_files:
                latest_batch = sorted(batch_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
                batch_path = os.path.join(self.checkpoint_dir, latest_batch)
                print(f"[Client {self.cid}] Loading from batch checkpoint: {batch_path}")
                self.net = YOLO()
                checkpoint_data = torch.load(batch_path, map_location='cpu')
                self.net.model.load_state_dict(checkpoint_data['model_state_dict'])
                checkpoint_loaded = True
                self.resume_training = True
        
        # Try default YOLO weights
        elif os.path.exists(f"{self.checkpoint_dir}/weights/last.pt"):
            print(f"[Client {self.cid}] Loading from YOLO weights: {self.checkpoint_dir}/weights/last.pt")
            self.net = YOLO(f"{self.checkpoint_dir}/weights/last.pt")
            checkpoint_loaded = True
            self.resume_training = True
        
        # If no checkpoint found, start fresh
        if not checkpoint_loaded:
            print(f"[Client {self.cid}] No checkpoint found, starting fresh training")
            self.net = YOLO()
            self.resume_training = False  # Don't resume for fresh training
        
        # Move model to CPU and ensure it's on CPU
        self.net.model = self.net.model.to('cpu')
        print(f"[Client {self.cid}] Model loaded on device: {self.device}")
        print(f"[Client {self.cid}] Resume training: {self.resume_training}")

    def get_parameters(self):
        """Get relevant model parameters based on the strategy."""
        current_state_dict = self.net.model.state_dict()
        # Use the imported function
        backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)

        # Define strategy groups (same as in set_parameters) - Corrected lists
        backbone_strategies = [
            'FedAvg', 'FedBackboneAvg', 'FedBackboneHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedBackboneMedian', 'FedBackboneHeadMedian', 'FedBackboneNeckMedian'
        ]
        neck_strategies = [
            'FedAvg', 'FedNeckAvg', 'FedNeckHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedNeckMedian', 'FedNeckHeadMedian', 'FedBackboneNeckMedian'
        ]
        head_strategies = [
            'FedAvg', 'FedHeadAvg', 'FedNeckHeadAvg', 'FedBackboneHeadAvg',
            'FedMedian', 'FedHeadMedian', 'FedNeckHeadMedian', 'FedBackboneHeadMedian'
        ]

        # Determine which parts to send based on strategy
        send_backbone = self.strategy_name in backbone_strategies
        send_neck = self.strategy_name in neck_strategies
        send_head = self.strategy_name in head_strategies

        relevant_parameters = []
        for k, v in current_state_dict.items():
            if (send_backbone and k in backbone_weights) or \
               (send_neck and k in neck_weights) or \
               (send_head and k in head_weights):
                relevant_parameters.append(v.cpu().numpy())
        
        return relevant_parameters

    def set_parameters(self, parameters):
        # Get current client model state and split into sections
        current_state_dict = self.net.model.state_dict()
        backbone_weights, neck_weights, head_weights = get_section_parameters(current_state_dict)
        
        # Define strategy groups
        backbone_strategies = [
            'FedAvg', 'FedBackboneAvg', 'FedBackboneHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedBackboneMedian', 'FedBackboneHeadMedian', 'FedBackboneNeckMedian'
        ]
        neck_strategies = [
            'FedAvg', 'FedNeckAvg', 'FedNeckHeadAvg', 'FedBackboneNeckAvg',
            'FedMedian', 'FedNeckMedian', 'FedNeckHeadMedian', 'FedBackboneNeckMedian'
        ]
        head_strategies = [
            'FedAvg', 'FedHeadAvg', 'FedNeckHeadAvg', 'FedBackboneHeadAvg',
            'FedMedian', 'FedHeadMedian', 'FedNeckHeadMedian', 'FedBackboneHeadMedian'
        ]
        
        # Determine which parts to update
        update_backbone = self.strategy_name in backbone_strategies
        update_neck = self.strategy_name in neck_strategies
        update_head = self.strategy_name in head_strategies
        
        # Create the SAME key order that get_parameters() used
        relevant_keys = []
        for k in current_state_dict.keys():
            if (update_backbone and k in backbone_weights) or \
            (update_neck and k in neck_weights) or \
            (update_head and k in head_weights):
                relevant_keys.append(k)
        
        # Verify parameter count matches
        if len(parameters) != len(relevant_keys):
            print(f"ERROR: Expected {len(relevant_keys)} parameters, got {len(parameters)}")
            return
        
        # NOW zip with the correct keys
        params_dict = zip(relevant_keys, parameters)
        
        # Apply the parameters
        updated_weights = {}
        for k, v in params_dict:
            updated_weights[k] = torch.tensor(v)

        updated_state_dict = OrderedDict(updated_weights)
        self.net.model.load_state_dict(updated_state_dict, strict=False)

    def fit(self, parameters, config):
        """Train the model using the provided parameters."""
        print(f"[Client {self.cid}] Starting training...")
        
        # Monitor memory before training
        print(f"[Client {self.cid}] Memory usage before training:")
        get_memory_usage()
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Train the model
        train(self.net, self.data_path, self.cid, self.strategy_name, self.checkpoint_dir, self.resume_training)
        
        # Monitor memory after training
        print(f"[Client {self.cid}] Memory usage after training:")
        get_memory_usage()
        
        # Clean up memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get updated parameters
        updated_parameters = self.get_parameters()
        
        # Clean up memory again
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[Client {self.cid}] Training completed.")
        return updated_parameters, len(self.data_path), {}


def main():

    args = parser.parse_args()
    assert args.cid < NUM_CLIENTS
    fl.client.start_client(server_address=SERVER_CONFIG['server_address'], 
                           client=FlowerClient(args.cid, args.data_path, SPLITS_CONFIG['dataset_name'], SERVER_CONFIG['strategy']))

if __name__ == "__main__":
    main()
    
