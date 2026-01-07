import torch
import polars as pl
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def eval_model_performance(
    y_actual: torch.Tensor, 
    y_pred: torch.Tensor, 
    feature_names: List[str], 
    target_name: str, 
    annualized_rate: float = np.sqrt(365 * 24 * 60)  # Assuming 15-minute data
) -> Dict[str, Any]:
    """
    Calculate performance metrics for the trading model.
    
    Parameters:
        y_actual (torch.Tensor): Actual target values.
        y_pred (torch.Tensor): Predicted target values.
        feature_names (List[str]): Names of features used in training.
        target_name (str): Name of the target variable.
        annualized_rate (float): Annualization factor for Sharpe ratio.
        
    Returns:
        Dict[str, Any]: Dictionary containing performance metrics.
    """
    # Convert tensors to numpy arrays
    y_actual_np = y_actual.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # Calculate basic regression metrics
    mse = mean_squared_error(y_actual_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual_np, y_pred_np)
    
    # Convert to Polars DataFrame for further processing if needed
    df = pl.DataFrame({
        "actual": y_actual_np.flatten(),
        "predicted": y_pred_np.flatten()
    })
    
    # Directional metrics (if applicable)
    actual_direction = np.sign(df['actual'])
    predicted_direction = np.sign(df['predicted'])
    
    # Accuracy based on direction prediction
    accuracy = accuracy_score(actual_direction, predicted_direction)
    
    return {
        'features': ','.join(feature_names),
        'target': target_name,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'direction_accuracy': accuracy
    }

def predict(model: torch.nn.Module, input_data: torch.Tensor) -> torch.Tensor:
    """
    Make a prediction using the trained model.
    
    Parameters:
        model (torch.nn.Module): Trained model.
        input_data (torch dataloader): Input tensor of shape [batchsize , 1, window_size, num_features].
        
    Returns:
        torch.Tensor: Prediction from the model.
    """
    model.eval()
    predictions = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_x, batch_y in input_data:
            batch_x = batch_x.to(device)
            batch_pred = model(batch_x)
            predictions.append(batch_pred)
    return torch.cat(predictions, dim=0).squeeze()

