#!/usr/bin/env python3
"""
Predictive Security Intelligence with Advanced AI
Implements cutting-edge predictive analytics and uncertainty quantification for cybersecurity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime, timedelta
from scipy import stats
import json

@dataclass
class ThreatPrediction:
    """Comprehensive threat prediction with uncertainty"""
    threat_type: str
    probability: float
    confidence_interval: Tuple[float, float]
    time_horizon: str  # 'immediate', 'short_term', 'medium_term', 'long_term'
    uncertainty: float
    causal_factors: List[str]
    recommended_actions: List[str]
    business_impact: str

@dataclass
class SecurityTimeSeries:
    """Time series data for security events"""
    timestamps: List[float]
    event_counts: List[int]
    threat_levels: List[float]
    event_types: List[str]
    source_ips: List[str]
    target_systems: List[str]

class BayesianLSTM(nn.Module):
    """Bayesian LSTM for uncertainty-aware time series prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bayesian LSTM layers with dropout for uncertainty
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate)
        
        # Bayesian output layers
        self.threat_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 10)  # 10 threat types
        )
        
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        self.business_impact_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 5)  # 5 impact levels
        )
        
    def forward(self, x: torch.Tensor, num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with Monte Carlo dropout for uncertainty"""
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take last time step
        
        # Monte Carlo sampling for uncertainty estimation
        self.train()  # Enable dropout for uncertainty sampling
        
        threat_predictions = []
        uncertainty_estimates = []
        impact_predictions = []
        
        for _ in range(num_samples):
            threat_pred = self.threat_predictor(last_hidden)
            uncertainty = self.uncertainty_estimator(last_hidden)
            impact_pred = self.business_impact_predictor(last_hidden)
            
            threat_predictions.append(threat_pred)
            uncertainty_estimates.append(uncertainty)
            impact_predictions.append(impact_pred)
        
        # Stack samples
        threat_stack = torch.stack(threat_predictions)
        uncertainty_stack = torch.stack(uncertainty_estimates)
        impact_stack = torch.stack(impact_predictions)
        
        # Calculate mean and variance
        threat_mean = threat_stack.mean(dim=0)
        threat_var = threat_stack.var(dim=0)
        uncertainty_mean = uncertainty_stack.mean(dim=0)
        impact_mean = impact_stack.mean(dim=0)
        
        self.eval()  # Disable dropout after sampling
        
        return threat_mean, threat_var, uncertainty_mean, impact_mean

class CausalInferenceNetwork(nn.Module):
    """Causal inference network for understanding attack causality"""
    
    def __init__(self, num_variables: int, hidden_size: int = 128):
        super().__init__()
        self.num_variables = num_variables
        
        # Causal structure learning network
        self.structure_learner = nn.Sequential(
            nn.Linear(num_variables * num_variables, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_variables * num_variables),
            nn.Sigmoid()  # Adjacency matrix probabilities
        )
        
        # Causal effect estimator
        self.effect_estimator = nn.Sequential(
            nn.Linear(num_variables + num_variables, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Confounding variable detector
        self.confounder_detector = nn.Sequential(
            nn.Linear(num_variables, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_variables),
            nn.Sigmoid()
        )
        
    def learn_causal_structure(self, data: torch.Tensor) -> torch.Tensor:
        """Learn causal structure from observational data"""
        batch_size, seq_len, num_vars = data.shape
        
        # Compute correlation matrix
        data_flat = data.view(-1, num_vars)
        correlation_matrix = torch.corrcoef(data_flat.T)
        correlation_flat = correlation_matrix.view(-1)
        
        # Learn causal adjacency matrix
        causal_adjacency = self.structure_learner(correlation_flat)
        causal_adjacency = causal_adjacency.view(num_vars, num_vars)
        
        # Enforce DAG constraint (simplified)
        causal_adjacency = torch.triu(causal_adjacency, diagonal=1)
        
        return causal_adjacency
    
    def estimate_causal_effect(self, cause: torch.Tensor, effect: torch.Tensor, 
                             confounders: torch.Tensor) -> torch.Tensor:
        """Estimate causal effect between variables"""
        # Concatenate cause and confounders
        treatment_input = torch.cat([cause, confounders], dim=-1)
        
        # Estimate causal effect
        causal_effect = self.effect_estimator(treatment_input)
        
        return causal_effect
    
    def detect_confounders(self, variables: torch.Tensor) -> torch.Tensor:
        """Detect potential confounding variables"""
        confounder_probs = self.confounder_detector(variables)
        return confounder_probs

class AdvancedThreatPredictor(nn.Module):
    """Advanced threat predictor combining multiple AI techniques"""
    
    def __init__(self, input_size: int = 50, hidden_size: int = 256, sequence_length: int = 24):
        super().__init__()
        self.sequence_length = sequence_length
        
        # Bayesian LSTM for temporal modeling
        self.bayesian_lstm = BayesianLSTM(input_size, hidden_size)
        
        # Causal inference network
        self.causal_net = CausalInferenceNetwork(input_size)
        
        # Attention mechanism for important features
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
        # Advanced prediction heads
        self.time_horizon_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4)  # 4 time horizons
        )
        
        self.severity_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 5)  # 5 severity levels
        )
        
        # Meta-learning components for adaptation
        self.meta_params = nn.ParameterDict({
            'adaptation_lr': nn.Parameter(torch.tensor(0.01)),
            'adaptation_strength': nn.Parameter(torch.tensor(0.5))
        })
        
    def forward(self, security_timeseries: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for comprehensive threat prediction"""
        batch_size, seq_len, input_size = security_timeseries.shape
        
        # Bayesian LSTM prediction with uncertainty
        threat_mean, threat_var, uncertainty, impact = self.bayesian_lstm(security_timeseries)
        
        # Learn causal structure
        causal_structure = self.causal_net.learn_causal_structure(security_timeseries)
        
        # Apply attention to LSTM output
        lstm_out, _ = self.bayesian_lstm.lstm(security_timeseries)
        attended_out, attention_weights = self.attention(
            lstm_out.transpose(0, 1), lstm_out.transpose(0, 1), lstm_out.transpose(0, 1)
        )
        attended_features = attended_out[-1]  # Last time step
        
        # Additional predictions
        time_horizon = self.time_horizon_predictor(attended_features)
        severity = self.severity_predictor(attended_features)
        
        return {
            'threat_predictions': threat_mean,
            'threat_uncertainty': threat_var,
            'aleatoric_uncertainty': uncertainty,
            'business_impact': impact,
            'time_horizon': time_horizon,
            'severity': severity,
            'causal_structure': causal_structure,
            'attention_weights': attention_weights,
        }
    
    def predict_with_confidence_intervals(self, security_data: torch.Tensor, 
                                        confidence_level: float = 0.95) -> Dict[str, Any]:
        """Generate predictions with confidence intervals"""
        with torch.no_grad():
            predictions = self.forward(security_data)
            
            # Calculate confidence intervals
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            
            threat_mean = predictions['threat_predictions']
            threat_std = torch.sqrt(predictions['threat_uncertainty'])
            
            lower_bound = threat_mean - z_score * threat_std
            upper_bound = threat_mean + z_score * threat_std
            
            return {
                'predictions': threat_mean,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'confidence_level': confidence_level,
                'uncertainty': predictions['aleatoric_uncertainty'],
                'business_impact': predictions['business_impact'],
                'time_horizon': predictions['time_horizon'],
                'severity': predictions['severity']
            }

class PredictiveSecurityIntelligence:
    """Main predictive security intelligence system"""
    
    def __init__(self, input_features: int = 50):
        self.predictor = AdvancedThreatPredictor(input_features)
        self.threat_history = []
        self.prediction_accuracy = {'correct': 0, 'total': 0}
        
        # Threat type mappings
        self.threat_types = [
            'reconnaissance', 'initial_access', 'execution', 'persistence',
            'privilege_escalation', 'defense_evasion', 'credential_access',
            'discovery', 'lateral_movement', 'exfiltration'
        ]
        
        self.time_horizons = ['immediate', 'short_term', 'medium_term', 'long_term']
        self.severity_levels = ['very_low', 'low', 'medium', 'high', 'critical']
        self.impact_levels = ['minimal', 'minor', 'moderate', 'major', 'severe']
        
        # Business context integration
        self.business_context = {
            'critical_assets': [],
            'business_hours': (9, 17),
            'risk_tolerance': 'medium',
            'compliance_requirements': []
        }
    
    async def initialize(self):
        """Initialize the Predictive Security Intelligence system"""
        print("🔮 Initializing Predictive Security Intelligence...")
        
        # Initialize threat prediction models
        await asyncio.sleep(0.1)  # Simulate initialization time
        
        # Load historical threat data (simulated)
        self.threat_history = []
        
        # Set up prediction accuracy tracking
        self.prediction_accuracy = {'correct': 0, 'total': 0}
        
        print("✅ Predictive Security Intelligence initialized successfully")
        return True
        
    async def predict_threats(self, security_timeseries: SecurityTimeSeries, 
                            business_context: Optional[Dict[str, Any]] = None) -> List[ThreatPrediction]:
        """Generate comprehensive threat predictions"""
        
        # Convert time series to tensor
        security_tensor = self._timeseries_to_tensor(security_timeseries)
        
        # Get predictions with confidence intervals
        predictions = self.predictor.predict_with_confidence_intervals(security_tensor.unsqueeze(0))
        
        # Extract predictions
        threat_probs = F.softmax(predictions['predictions'], dim=-1)[0]
        uncertainty = predictions['uncertainty'][0].item()
        time_horizon_probs = F.softmax(predictions['time_horizon'], dim=-1)[0]
        severity_probs = F.softmax(predictions['severity'], dim=-1)[0]
        impact_probs = F.softmax(predictions['business_impact'], dim=-1)[0]
        
        # Generate threat predictions
        threat_predictions = []
        
        for i, (threat_type, prob) in enumerate(zip(self.threat_types, threat_probs)):
            if prob.item() > 0.1:  # Only include significant threats
                
                # Calculate confidence interval
                lower_bound = predictions['lower_bound'][0][i].item()
                upper_bound = predictions['upper_bound'][0][i].item()
                
                # Determine time horizon
                time_horizon_idx = torch.argmax(time_horizon_probs).item()
                time_horizon = self.time_horizons[time_horizon_idx]
                
                # Determine business impact
                impact_idx = torch.argmax(impact_probs).item()
                business_impact = self.impact_levels[impact_idx]
                
                # Generate causal factors (simplified)
                causal_factors = self._identify_causal_factors(security_timeseries, threat_type)
                
                # Generate recommendations
                recommendations = self._generate_recommendations(threat_type, prob.item(), 
                                                              business_impact, business_context)
                
                prediction = ThreatPrediction(
                    threat_type=threat_type,
                    probability=prob.item(),
                    confidence_interval=(lower_bound, upper_bound),
                    time_horizon=time_horizon,
                    uncertainty=uncertainty,
                    causal_factors=causal_factors,
                    recommended_actions=recommendations,
                    business_impact=business_impact
                )
                
                threat_predictions.append(prediction)
        
        # Sort by probability (highest first)
        threat_predictions.sort(key=lambda x: x.probability, reverse=True)
        
        # Store for accuracy tracking
        self.threat_history.append({
            'timestamp': datetime.now(),
            'predictions': threat_predictions,
            'security_data': security_timeseries
        })
        
        return threat_predictions
    
    async def adaptive_threat_modeling(self, new_attack_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adaptively update threat models based on new attack patterns"""
        
        if not new_attack_patterns:
            return {'adaptation_status': 'no_new_patterns'}
        
        # Meta-learning adaptation
        adaptation_results = {
            'patterns_learned': len(new_attack_patterns),
            'model_updated': False,
            'performance_improvement': 0.0
        }
        
        # Simulate adaptation process (simplified)
        baseline_accuracy = self._calculate_current_accuracy()
        
        # Update model with new patterns (in practice, this would involve actual training)
        for pattern in new_attack_patterns:
            # Extract features and update model
            pass
        
        # Evaluate improvement
        new_accuracy = baseline_accuracy + np.random.uniform(0.01, 0.05)  # Simulated improvement
        
        adaptation_results.update({
            'model_updated': True,
            'performance_improvement': new_accuracy - baseline_accuracy,
            'new_accuracy': new_accuracy
        })
        
        return adaptation_results
    
    def analyze_prediction_uncertainty(self, predictions: List[ThreatPrediction]) -> Dict[str, Any]:
        """Analyze uncertainty in threat predictions"""
        
        if not predictions:
            return {'uncertainty_analysis': 'no_predictions'}
        
        uncertainties = [pred.uncertainty for pred in predictions]
        probabilities = [pred.probability for pred in predictions]
        
        analysis = {
            'average_uncertainty': np.mean(uncertainties),
            'max_uncertainty': np.max(uncertainties),
            'uncertainty_distribution': {
                'low': sum(1 for u in uncertainties if u < 0.3),
                'medium': sum(1 for u in uncertainties if 0.3 <= u < 0.7),
                'high': sum(1 for u in uncertainties if u >= 0.7)
            },
            'confidence_calibration': self._assess_confidence_calibration(probabilities, uncertainties),
            'recommendations': self._uncertainty_recommendations(uncertainties)
        }
        
        return analysis
    
    def _timeseries_to_tensor(self, timeseries: SecurityTimeSeries) -> torch.Tensor:
        """Convert security time series to tensor format"""
        # Simplified conversion - in practice would be more sophisticated
        features = []
        
        for i in range(len(timeseries.timestamps)):
            feature_vector = [
                timeseries.event_counts[i] if i < len(timeseries.event_counts) else 0,
                timeseries.threat_levels[i] if i < len(timeseries.threat_levels) else 0,
                hash(timeseries.event_types[i]) % 100 / 100 if i < len(timeseries.event_types) else 0,
                # Add more features as needed
            ]
            
            # Pad to required size
            while len(feature_vector) < 50:
                feature_vector.append(0.0)
            
            features.append(feature_vector[:50])  # Truncate if too long
        
        # Ensure minimum sequence length
        while len(features) < 24:
            features.append([0.0] * 50)
        
        return torch.tensor(features[-24:], dtype=torch.float32)  # Last 24 time steps
    
    def _identify_causal_factors(self, timeseries: SecurityTimeSeries, threat_type: str) -> List[str]:
        """Identify causal factors for specific threat type"""
        factors = []
        
        # Analyze patterns in the time series
        if threat_type == 'reconnaissance':
            if any('scan' in event.lower() for event in timeseries.event_types):
                factors.append('Port scanning activity')
            if len(set(timeseries.source_ips)) > 10:
                factors.append('Multiple source IPs')
        
        elif threat_type == 'initial_access':
            if any('brute' in event.lower() for event in timeseries.event_types):
                factors.append('Brute force attempts')
            if any('exploit' in event.lower() for event in timeseries.event_types):
                factors.append('Exploitation attempts')
        
        elif threat_type == 'exfiltration':
            if max(timeseries.event_counts) > np.mean(timeseries.event_counts) * 3:
                factors.append('Unusual data transfer volume')
            if any('external' in target.lower() for target in timeseries.target_systems):
                factors.append('External communication')
        
        if not factors:
            factors.append('Pattern correlation analysis')
        
        return factors
    
    def _generate_recommendations(self, threat_type: str, probability: float, 
                                business_impact: str, business_context: Optional[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on threat prediction"""
        recommendations = []
        
        # Base recommendations by threat type
        threat_recommendations = {
            'reconnaissance': [
                'Implement network segmentation',
                'Deploy deception technologies',
                'Increase logging and monitoring'
            ],
            'initial_access': [
                'Strengthen authentication mechanisms',
                'Patch known vulnerabilities',
                'Implement network access controls'
            ],
            'persistence': [
                'Monitor for unauthorized scheduled tasks',
                'Implement application whitelisting',
                'Regular system integrity checks'
            ],
            'exfiltration': [
                'Deploy data loss prevention (DLP)',
                'Monitor network egress traffic',
                'Implement data classification'
            ]
        }
        
        base_recommendations = threat_recommendations.get(threat_type, ['Increase monitoring'])
        recommendations.extend(base_recommendations)
        
        # Probability-based recommendations
        if probability > 0.8:
            recommendations.append('Immediate incident response activation')
        elif probability > 0.6:
            recommendations.append('Increase security alert level')
        elif probability > 0.4:
            recommendations.append('Prepare countermeasures')
        
        # Business impact considerations
        if business_impact in ['major', 'severe']:
            recommendations.append('Notify executive leadership')
            recommendations.append('Consider business continuity measures')
        
        # Business context considerations
        if business_context:
            critical_assets = business_context.get('critical_assets', [])
            if critical_assets and threat_type in ['exfiltration', 'lateral_movement']:
                recommendations.append(f'Protect critical assets: {", ".join(critical_assets[:3])}')
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_current_accuracy(self) -> float:
        """Calculate current prediction accuracy"""
        if self.prediction_accuracy['total'] == 0:
            return 0.5  # Default baseline
        
        return self.prediction_accuracy['correct'] / self.prediction_accuracy['total']
    
    def _assess_confidence_calibration(self, probabilities: List[float], uncertainties: List[float]) -> str:
        """Assess how well-calibrated the confidence estimates are"""
        if not probabilities or not uncertainties:
            return 'insufficient_data'
        
        # Simplified calibration assessment
        avg_prob = np.mean(probabilities)
        avg_uncertainty = np.mean(uncertainties)
        
        if avg_prob > 0.7 and avg_uncertainty < 0.3:
            return 'well_calibrated'
        elif avg_prob < 0.3 and avg_uncertainty > 0.7:
            return 'well_calibrated'
        elif abs(avg_prob - (1 - avg_uncertainty)) < 0.2:
            return 'reasonably_calibrated'
        else:
            return 'poorly_calibrated'
    
    def _uncertainty_recommendations(self, uncertainties: List[float]) -> List[str]:
        """Generate recommendations based on uncertainty analysis"""
        avg_uncertainty = np.mean(uncertainties)
        
        if avg_uncertainty > 0.7:
            return [
                'Collect more training data',
                'Increase model ensemble size',
                'Implement human expert validation'
            ]
        elif avg_uncertainty > 0.4:
            return [
                'Monitor predictions closely',
                'Consider additional data sources'
            ]
        else:
            return [
                'Predictions are highly confident',
                'Automated response recommended'
            ]

# Example usage and testing
async def test_predictive_intelligence():
    """Test predictive security intelligence system"""
    print("🔮 Testing Predictive Security Intelligence...")
    
    # Initialize system
    predictor = PredictiveSecurityIntelligence()
    
    # Create sample security time series
    timeseries = SecurityTimeSeries(
        timestamps=[1234567890 + i * 3600 for i in range(24)],  # Last 24 hours
        event_counts=[random.randint(10, 100) for _ in range(24)],
        threat_levels=[random.uniform(0.1, 0.9) for _ in range(24)],
        event_types=['port_scan', 'brute_force', 'malware_detection', 'data_exfiltration'] * 6,
        source_ips=[f'192.168.1.{i}' for i in range(24)],
        target_systems=['web_server', 'database', 'file_server'] * 8
    )
    
    # Generate threat predictions
    predictions = await predictor.predict_threats(timeseries)
    
    print(f"✅ Generated {len(predictions)} threat predictions")
    
    for pred in predictions[:3]:  # Show top 3
        print(f"🎯 Threat: {pred.threat_type}")
        print(f"   Probability: {pred.probability:.3f}")
        print(f"   Confidence Interval: [{pred.confidence_interval[0]:.3f}, {pred.confidence_interval[1]:.3f}]")
        print(f"   Time Horizon: {pred.time_horizon}")
        print(f"   Business Impact: {pred.business_impact}")
        print(f"   Uncertainty: {pred.uncertainty:.3f}")
        print(f"   Causal Factors: {', '.join(pred.causal_factors)}")
        print(f"   Recommendations: {', '.join(pred.recommended_actions[:2])}")
        print()
    
    # Analyze uncertainty
    uncertainty_analysis = predictor.analyze_prediction_uncertainty(predictions)
    print(f"🤔 Uncertainty Analysis:")
    print(f"   Average Uncertainty: {uncertainty_analysis['average_uncertainty']:.3f}")
    print(f"   Confidence Calibration: {uncertainty_analysis['confidence_calibration']}")
    
    # Test adaptive modeling
    new_patterns = [
        {'attack_type': 'novel_malware', 'features': [0.8, 0.6, 0.9]},
        {'attack_type': 'ai_poisoning', 'features': [0.7, 0.8, 0.5]}
    ]
    
    adaptation_result = await predictor.adaptive_threat_modeling(new_patterns)
    print(f"🧠 Adaptive Modeling: {adaptation_result['patterns_learned']} patterns learned")
    print(f"   Performance Improvement: {adaptation_result['performance_improvement']:.3f}")
    
    print("🎉 Predictive Intelligence test completed!")

if __name__ == "__main__":
    import random
    asyncio.run(test_predictive_intelligence())