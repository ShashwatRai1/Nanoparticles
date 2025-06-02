import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, LSTM, GRU
from keras.optimizers import Adam, AdamW
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Lambda
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import os, seaborn as sns
import pickle
import joblib
import json

print(f"TensorFlow: {tf.__version__}, Keras: {keras.__version__}")
np.random.seed(42); tf.random.set_seed(42)

class EnhancedNanophotonicModel:
    def __init__(self, data_path):
        self.data_path, self.model_dir = data_path, "enhanced_models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.range_models, self.neural_models, self.scalers, self.ensemble_models = self._define_range_models(), {}, {}, {}
        
    def _define_range_models(self):
        return {
            (0, 50): {'wavelength': lambda d, n: 322.17 + 1.048*d + 396.93*n - 0.00196*d**2 - 1.933*d*n - 293.84*n**2 - 0.0000534*d**3 + 0.00643*d**2*n + 0.841*d*n**2 + 81.14*n**3,
                     'extinction': lambda d, n: 4011.96 + 712.15*d - 15418.9*n - 14.914*d**2 - 895.34*d*n + 15960.09*n**2 + 0.0695*d**3 + 10.973*d**2*n + 267.04*d*n**2 - 4953.15*n**3},
            (51, 100): {'wavelength': lambda d, n: 616.64 - 0.4999*d - 150.88*n + 0.00796*d**2 - 1.632*d*n + 81.67*n**2 - 0.0000326*d**3 + 0.00386*d**2*n + 1.105*d*n**2 - 10.087*n**3,
                       'extinction': lambda d, n: 356419.94 - 8415.48*d - 422956.8*n + 44.909*d**2 + 7854.24*d*n + 131133.01*n**2 - 0.0874*d**3 - 15.183*d**2*n - 1875.27*d*n**2 - 2180.07*n**3},
            (101, 150): {'wavelength': lambda d, n: -316.64 + 56.428*d - 3760.25*n - 0.351*d**2 - 22.939*d*n + 4265.47*n**2 + 0.000577*d**3 + 0.117*d**2*n - 1.798*d*n**2 - 1085.24*n**3,
                        'extinction': lambda d, n: -672229.58 + 8951.1*d + 321988.41*n - 43.331*d**2 - 2598.45*d*n + 83122.05*n**2 + 0.0857*d**3 + 7.547*d**2*n - 236.24*d*n**2 - 46971.63*n**3},
            (151, 200): {'wavelength': lambda d, n: 699.38 + 4.646*d - 894.36*n - 0.0152*d**2 - 6.634*d*n + 998.7*n**2 + 0.00000315*d**3 + 0.0177*d**2*n + 1.699*d*n**2 - 286.87*n**3,
                        'extinction': lambda d, n: -1410296.94 + 10843.77*d + 1850112.5*n - 22.053*d**2 - 9334.12*d*n - 781174.07*n**2 + 0.0198*d**3 + 8.904*d**2*n + 2216.51*d*n**2 + 99660.63*n**3},
            (201, 250): {'wavelength': lambda d, n: 5355.24 - 41.275*d - 4359.36*n + 0.0947*d**2 + 28.462*d*n + 963.67*n**2 - 0.0000496*d**3 - 0.0406*d**2*n - 2.502*d*n**2 - 84.871*n**3,
                        'extinction': lambda d, n: -155298.03 + 4343.88*d - 109987.67*n - 12.754*d**2 - 1959.08*d*n + 179483.12*n**2 + 0.0136*d**3 + 4.201*d**2*n + 100.02*d*n**2 - 41765.12*n**3},
            (251, 300): {'wavelength': lambda d, n: 1541.24 - 6.795*d - 2378.29*n + 0.0124*d**2 + 8.424*d*n + 1379.81*n**2 - 0.00000806*d**3 - 0.00506*d**2*n - 1.459*d*n**2 - 269.76*n**3,
                        'extinction': lambda d, n: 973030.22 - 4906.66*d - 1000387.44*n + 12.528*d**2 + 2875.81*d*n + 418215.95*n**2 - 0.00885*d**3 - 2.642*d**2*n - 502.05*d*n**2 - 65392.55*n**3}
        }
    
    def load_data(self):
        sheet_to_ri, all_data = {'1': 1.0, '2': 1.2, '3': 1.3, '4': 1.33, '5': 1.4, '6': 1.5}, []
        for sheet_name, ri in sheet_to_ri.items():
            try:
                df = pd.read_excel(self.data_path, sheet_name=sheet_name)
                df.columns = [col.strip().lower() for col in df.columns]
                column_mapping = {'size (diameter)': 'diameter', 'refractive index of surrounding': 'refractive_index', 'resonance wavelength': 'resonance_wavelength', 'resonance intensity (extinction)': 'extinction'}
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                if 'refractive_index' not in df.columns: df['refractive_index'] = ri
                all_data.append(df)
                print(f"Loaded sheet {sheet_name} with RI {ri}")
            except Exception as e: print(f"Error loading sheet {sheet_name}: {e}")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_data)} data points")
        if combined_data.isna().any().any() or np.isinf(combined_data.values).any():
            print("Cleaning NaN/infinite values...")
            combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna()
            print(f"After cleaning: {len(combined_data)} data points")
        return combined_data
    
    def find_diameter_range(self, diameter):
        for (min_d, max_d) in self.range_models.keys():
            if min_d <= diameter <= max_d: return (min_d, max_d)
        closest_range = min(self.range_models.keys(), key=lambda x: min(abs(diameter - x[0]), abs(diameter - x[1])))
        print(f"Warning: Diameter {diameter} outside ranges. Using {closest_range}.")
        return closest_range
    
    def predict_with_range_formulas(self, diameter, refractive_index):
        range_key = self.find_diameter_range(diameter)
        range_model = self.range_models[range_key]
        return range_model['wavelength'](diameter, refractive_index), range_model['extinction'](diameter, refractive_index)
    
    def calculate_enhanced_features(self, df):
        diameter, ri = df['diameter'], df['refractive_index']
        features = pd.DataFrame()
        
        features['diameter'], features['refractive_index'] = diameter, ri
        features['diameter_squared'], features['diameter_cubed'] = diameter ** 2, diameter ** 3
        features['ri_squared'], features['ri_cubed'] = ri ** 2, ri ** 3
        features['d_times_ri'], features['d_squared_times_ri'] = diameter * ri, diameter ** 2 * ri
        features['d_times_ri_squared'], features['d_cubed_times_ri'] = diameter * ri ** 2, diameter ** 3 * ri
        features['d_squared_times_ri_squared'], features['d_times_ri_cubed'] = diameter ** 2 * ri ** 2, diameter * ri ** 3
        
        theoretical_wavelengths, theoretical_extinctions = [], []
        for i in range(len(diameter)):
            d, n = diameter.iloc[i], ri.iloc[i]
            theoretical_wavelength, theoretical_extinction = self.predict_with_range_formulas(d, n)
            theoretical_wavelengths.append(theoretical_wavelength)
            theoretical_extinctions.append(theoretical_extinction)
        
        features['theoretical_wavelength'], features['theoretical_extinction'] = theoretical_wavelengths, theoretical_extinctions
        features['size_parameter'] = np.pi * diameter / features['theoretical_wavelength']
        features['optical_path'], features['relative_refractive_index'] = diameter * ri, ri / 1.33
        features['quality_factor'] = features['theoretical_wavelength'] / (features['theoretical_wavelength'] * 0.05)
        features['mie_parameter'] = 2 * np.pi * diameter * ri / features['theoretical_wavelength']
        features['sin_mie_param'], features['cos_mie_param'] = np.sin(features['mie_parameter']), np.cos(features['mie_parameter'])
        
        features['log_diameter'], features['log_ri'] = np.log(diameter + 1), np.log(ri)
        features['sqrt_diameter'], features['sqrt_ri'] = np.sqrt(diameter), np.sqrt(ri)
        features['exp_d_ri'] = np.exp(-diameter * ri / 100)
        features['tanh_mie'] = np.tanh(features['mie_parameter'])
        features['volume'] = (4/3) * np.pi * (diameter/2)**3
        features['surface_area'] = 4 * np.pi * (diameter/2)**2
        features['aspect_ratio'] = features['volume'] / features['surface_area']
        
        return features
    
    def train_ensemble_models(self, df):
        """Train ensemble models (Random Forest and Gradient Boosting) for each diameter range"""
        print("Training ensemble models...")
        
        for range_key in self.range_models.keys():
            print(f"Training models for range {range_key}")
            
            # Filter data for this range
            mask = (df['diameter'] >= range_key[0]) & (df['diameter'] <= range_key[1])
            range_data = df[mask].copy()
            
            if len(range_data) < 10:
                print(f"Insufficient data for range {range_key} ({len(range_data)} samples)")
                continue
            
            # Calculate features
            X = self.calculate_enhanced_features(range_data)
            y = range_data[['resonance_wavelength', 'extinction']].values
            
            # Handle any remaining NaN values
            mask_valid = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
            X, y = X[mask_valid], y[mask_valid]
            
            if len(X) < 10:
                print(f"Insufficient valid data for range {range_key}")
                continue
            
            # Train Random Forest and Gradient Boosting
            rf_model = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            )
            gb_model = MultiOutputRegressor(
                GradientBoostingRegressor(n_estimators=100, random_state=42)
            )
            
            rf_model.fit(X, y)
            gb_model.fit(X, y)
            
            self.ensemble_models[range_key] = {
                'rf': rf_model,
                'gb': gb_model
            }
            
            print(f"Range {range_key}: Trained with {len(X)} samples")
    
    def create_neural_model(self, input_dim):
        """Create a neural network model architecture"""
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dense(2, activation='linear')  # wavelength and extinction
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_neural_models(self, df):
        """Train neural network models for each diameter range"""
        print("Training neural network models...")
        
        for range_key in self.range_models.keys():
            print(f"Training neural model for range {range_key}")
            
            # Filter data for this range
            mask = (df['diameter'] >= range_key[0]) & (df['diameter'] <= range_key[1])
            range_data = df[mask].copy()
            
            if len(range_data) < 20:
                print(f"Insufficient data for neural model in range {range_key}")
                continue
            
            # Calculate features
            X = self.calculate_enhanced_features(range_data)
            y = range_data[['resonance_wavelength', 'extinction']].values
            
            # Handle any remaining NaN values
            mask_valid = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
            X, y = X[mask_valid], y[mask_valid]
            
            if len(X) < 20:
                print(f"Insufficient valid data for neural model in range {range_key}")
                continue
            
            # Scale features and targets
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            X_scaled = X_scaler.fit_transform(X)
            y_scaled = y_scaler.fit_transform(y)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            # Create and train model
            model = self.create_neural_model(X_scaled.shape[1])
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=50, restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=200,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Store model and scalers
            self.neural_models[range_key] = model
            self.scalers[range_key] = {
                'X_scaler': X_scaler,
                'y_scaler': y_scaler
            }
            
            # Print performance
            train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
            val_loss = model.evaluate(X_test, y_test, verbose=0)[0]
            print(f"Range {range_key}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def predict_ensemble(self, diameters, refractive_indices):
        if not isinstance(diameters, list): diameters = [diameters]
        if not isinstance(refractive_indices, list): refractive_indices = [refractive_indices] * len(diameters)
        
        results = {'formula_wavelength': [], 'formula_extinction': [], 'neural_wavelength': [], 'neural_extinction': [], 'rf_wavelength': [], 'rf_extinction': [], 'gb_wavelength': [], 'gb_extinction': [], 'ensemble_wavelength': [], 'ensemble_extinction': []}
        
        for d, n in zip(diameters, refractive_indices):
            range_key = self.find_diameter_range(d)
            formula_w, formula_e = self.predict_with_range_formulas(d, n)
            results['formula_wavelength'].append(formula_w)
            results['formula_extinction'].append(formula_e)
            
            df = pd.DataFrame({'diameter': [d], 'refractive_index': [n]})
            features = self.calculate_enhanced_features(df)
            
            neural_w, neural_e, rf_w, rf_e, gb_w, gb_e = None, None, None, None, None, None
            
            if range_key in self.neural_models and range_key in self.scalers:
                X_scaler, y_scaler, model = self.scalers[range_key]['X_scaler'], self.scalers[range_key]['y_scaler'], self.neural_models[range_key]
                X_scaled = X_scaler.transform(features.values)
                y_pred_scaled = model.predict(X_scaled, verbose=0)
                y_pred = y_scaler.inverse_transform(y_pred_scaled)
                neural_w, neural_e = y_pred[0]
            
            if range_key in self.ensemble_models:
                rf_pred = self.ensemble_models[range_key]['rf'].predict(features.values)
                gb_pred = self.ensemble_models[range_key]['gb'].predict(features.values)
                rf_w, rf_e = rf_pred[0]
                gb_w, gb_e = gb_pred[0]
            
            if all(x is not None for x in [neural_w, rf_w, gb_w]):
                ensemble_w = 0.5 * neural_w + 0.3 * rf_w + 0.2 * gb_w
                ensemble_e = 0.5 * neural_e + 0.3 * rf_e + 0.2 * gb_e
            else:
                ensemble_w, ensemble_e = neural_w or formula_w, neural_e or formula_e
            
            results['neural_wavelength'].append(neural_w)
            results['neural_extinction'].append(neural_e)
            results['rf_wavelength'].append(rf_w)
            results['rf_extinction'].append(rf_e)
            results['gb_wavelength'].append(gb_w)
            results['gb_extinction'].append(gb_e)
            results['ensemble_wavelength'].append(ensemble_w)
            results['ensemble_extinction'].append(ensemble_e)
        
        return pd.DataFrame(results)
    
    def save_complete_model(self, filename="complete_nanophotonic_model"):
        """Save all models, scalers, and metadata"""
        save_path = os.path.join(self.model_dir, filename)
        os.makedirs(save_path, exist_ok=True)
        
        # Save neural models (Keras models)
        neural_models_path = os.path.join(save_path, "neural_models")
        os.makedirs(neural_models_path, exist_ok=True)
        for range_key, model in self.neural_models.items():
            model_name = f"neural_{range_key[0]}_{range_key[1]}.keras"
            model.save(os.path.join(neural_models_path, model_name))
        
        # Save ensemble models (sklearn models)
        ensemble_models_path = os.path.join(save_path, "ensemble_models")
        os.makedirs(ensemble_models_path, exist_ok=True)
        for range_key, models in self.ensemble_models.items():
            range_name = f"{range_key[0]}_{range_key[1]}"
            joblib.dump(models['rf'], os.path.join(ensemble_models_path, f"rf_{range_name}.pkl"))
            joblib.dump(models['gb'], os.path.join(ensemble_models_path, f"gb_{range_name}.pkl"))
        
        # Save scalers
        scalers_path = os.path.join(save_path, "scalers")
        os.makedirs(scalers_path, exist_ok=True)
        for range_key, scalers in self.scalers.items():
            range_name = f"{range_key[0]}_{range_key[1]}"
            joblib.dump(scalers['X_scaler'], os.path.join(scalers_path, f"X_scaler_{range_name}.pkl"))
            joblib.dump(scalers['y_scaler'], os.path.join(scalers_path, f"y_scaler_{range_name}.pkl"))
        
        # Save metadata (range information)
        metadata = {
            'range_keys': [list(key) for key in self.range_models.keys()],
            'model_info': {
                'neural_models': list(self.neural_models.keys()),
                'ensemble_models': list(self.ensemble_models.keys()),
                'scalers': list(self.scalers.keys())
            }
        }
        with open(os.path.join(save_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Complete model saved to: {save_path}")
        return save_path
    
    def load_complete_model(self, filename="complete_nanophotonic_model"):
        """Load all models, scalers, and metadata"""
        load_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model directory not found: {load_path}")
        
        # Load metadata
        with open(os.path.join(load_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # Load neural models
        neural_models_path = os.path.join(load_path, "neural_models")
        self.neural_models = {}
        for range_key_list in metadata['range_keys']:
            range_key = tuple(range_key_list)
            model_name = f"neural_{range_key[0]}_{range_key[1]}.keras"
            model_path = os.path.join(neural_models_path, model_name)
            if os.path.exists(model_path):
                self.neural_models[range_key] = load_model(model_path)
        
        # Load ensemble models
        ensemble_models_path = os.path.join(load_path, "ensemble_models")
        self.ensemble_models = {}
        for range_key_list in metadata['range_keys']:
            range_key = tuple(range_key_list)
            range_name = f"{range_key[0]}_{range_key[1]}"
            rf_path = os.path.join(ensemble_models_path, f"rf_{range_name}.pkl")
            gb_path = os.path.join(ensemble_models_path, f"gb_{range_name}.pkl")
            
            if os.path.exists(rf_path) and os.path.exists(gb_path):
                self.ensemble_models[range_key] = {
                    'rf': joblib.load(rf_path),
                    'gb': joblib.load(gb_path)
                }
        
        # Load scalers
        scalers_path = os.path.join(load_path, "scalers")
        self.scalers = {}
        for range_key_list in metadata['range_keys']:
            range_key = tuple(range_key_list)
            range_name = f"{range_key[0]}_{range_key[1]}"
            x_scaler_path = os.path.join(scalers_path, f"X_scaler_{range_name}.pkl")
            y_scaler_path = os.path.join(scalers_path, f"y_scaler_{range_name}.pkl")
            
            if os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path):
                self.scalers[range_key] = {
                    'X_scaler': joblib.load(x_scaler_path),
                    'y_scaler': joblib.load(y_scaler_path)
                }
        
        print(f"Complete model loaded from: {load_path}")
        print(f"Loaded {len(self.neural_models)} neural models")
        print(f"Loaded {len(self.ensemble_models)} ensemble models")
        print(f"Loaded {len(self.scalers)} scaler sets")
    
    def interactive_prediction(self):
        print("\nEnhanced Interactive Prediction")
        print("=" * 40)
        while True:
            try:
                diameter = float(input("\nDiameter (nm) [-1 to exit]: "))
                if diameter == -1: break
                refractive_index = float(input("Refractive index: "))
                
                predictions = self.predict_ensemble(diameter, refractive_index)
                print(f"\nResults for D={diameter}nm, RI={refractive_index}:")
                print(f"Formula: λ={predictions['formula_wavelength'][0]:.2f}nm, E={predictions['formula_extinction'][0]:.2f}")
                if predictions['neural_wavelength'][0]: print(f"Neural:  λ={predictions['neural_wavelength'][0]:.2f}nm, E={predictions['neural_extinction'][0]:.2f}")
                if predictions['ensemble_wavelength'][0]: print(f"Ensemble: λ={predictions['ensemble_wavelength'][0]:.2f}nm, E={predictions['ensemble_extinction'][0]:.2f}")
                
            except ValueError: print("Invalid input. Enter numeric values.")
            except Exception as e: print(f"Error: {e}")
    
    def evaluate_model_performance(self, df):
        """Evaluate model performance on test data"""
        print("\nEvaluating model performance...")
        
        for range_key in self.range_models.keys():
            mask = (df['diameter'] >= range_key[0]) & (df['diameter'] <= range_key[1])
            range_data = df[mask].copy()
            
            if len(range_data) < 10:
                continue
                
            diameters = range_data['diameter'].tolist()
            ris = range_data['refractive_index'].tolist()
            actual_wavelengths = range_data['resonance_wavelength'].values
            actual_extinctions = range_data['extinction'].values
            
            predictions = self.predict_ensemble(diameters, ris)
            
            # Calculate metrics for wavelength
            if predictions['ensemble_wavelength'][0] is not None:
                ensemble_wavelengths = [x for x in predictions['ensemble_wavelength'] if x is not None]
                if len(ensemble_wavelengths) > 0:
                    mae_w = mean_absolute_error(actual_wavelengths, ensemble_wavelengths)
                    rmse_w = np.sqrt(mean_squared_error(actual_wavelengths, ensemble_wavelengths))
                    r2_w = r2_score(actual_wavelengths, ensemble_wavelengths)
                    
                    print(f"Range {range_key} - Wavelength:")
                    print(f"  MAE: {mae_w:.2f} nm")
                    print(f"  RMSE: {rmse_w:.2f} nm")
                    print(f"  R²: {r2_w:.3f}")


# Example usage scripts:

def train_and_save_model():
    """Train the model and save it"""
    print("Training and saving model...")
    model = EnhancedNanophotonicModel("resonancedata1.xlsx")
    
    # Load and train on data
    df = model.load_data()
    model.train_ensemble_models(df)
    model.train_neural_models(df)
    
    # Evaluate performance
    model.evaluate_model_performance(df)
    
    # Save the complete model
    save_path = model.save_complete_model("my_nanophotonic_model_v1")
    print(f"\nModel training complete and saved to: {save_path}")
    return model

def load_and_use_model():
    """Load a pre-trained model and use it for predictions"""
    print("Loading pre-trained model...")
    
    # Create model instance (no need to provide data_path for prediction only)
    model = EnhancedNanophotonicModel(data_path=None)
    
    # Load the trained model
    model.load_complete_model("my_nanophotonic_model_v1")
    
    # Make predictions
    test_diameters = [25, 75, 125, 175, 225]
    test_ris = [1.2, 1.3, 1.4, 1.5, 1.6]
    
    predictions = model.predict_ensemble(test_diameters, test_ris)
    print("\nPrediction Results:")
    print(predictions[['formula_wavelength', 'formula_extinction', 
                     'ensemble_wavelength', 'ensemble_extinction']])
    
    return model

def quick_prediction_script():
    """Quick script for making single predictions with loaded model"""
    model = EnhancedNanophotonicModel(data_path=None)
    model.load_complete_model("my_nanophotonic_model_v1")
    
    # Single prediction
    diameter = 100  # nm
    ri = 1.4
    
    prediction = model.predict_ensemble(diameter, ri)
    print(f"\nPrediction for D={diameter}nm, RI={ri}:")
    print(f"Wavelength: {prediction['ensemble_wavelength'][0]:.2f} nm")
    print(f"Extinction: {prediction['ensemble_extinction'][0]:.2f}")

def interactive_mode():
    """Run interactive prediction mode"""
    print("Starting interactive mode...")
    model = EnhancedNanophotonicModel(data_path=None)
    
    try:
        model.load_complete_model("my_nanophotonic_model_v1")
        model.interactive_prediction()
    except FileNotFoundError:
        print("No trained model found. Please run train_and_save_model() first.")

def batch_prediction_from_file(input_file, output_file):
    """Make batch predictions from a CSV file"""
    model = EnhancedNanophotonicModel(data_path=None)
    model.load_complete_model("my_nanophotonic_model_v1")
    
    # Load input data
    input_data = pd.read_csv(input_file)
    required_columns = ['diameter', 'refractive_index']
    
    if not all(col in input_data.columns for col in required_columns):
        raise ValueError(f"Input file must contain columns: {required_columns}")
    
    # Make predictions
    diameters = input_data['diameter'].tolist()
    ris = input_data['refractive_index'].tolist()
    
    predictions = model.predict_ensemble(diameters, ris)
    
    # Combine input and predictions
    result_data = pd.concat([input_data, predictions], axis=1)
    
    # Save results
    result_data.to_csv(output_file, index=False)
    print(f"Batch predictions saved to: {output_file}")
    
    return result_data

def create_sample_input_file():
    """Create a sample input file for batch predictions"""
    sample_data = pd.DataFrame({
        'diameter': [25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
        'refractive_index': [1.2, 1.3, 1.4, 1.5, 1.6, 1.2, 1.3, 1.4, 1.5, 1.6]
    })
    
    sample_data.to_csv('sample_input.csv', index=False)
    print("Sample input file created: sample_input.csv")
    return sample_data

def visualize_predictions():
    """Create visualizations of model predictions vs theoretical values"""
    model = EnhancedNanophotonicModel(data_path=None)
    model.load_complete_model("my_nanophotonic_model_v1")
    
    # Create test data
    diameters = np.linspace(10, 290, 50)
    ris = [1.2, 1.33, 1.5]
    
    plt.figure(figsize=(15, 10))
    
    for i, ri in enumerate(ris):
        predictions = model.predict_ensemble(diameters.tolist(), [ri] * len(diameters))
        
        plt.subplot(2, 3, i + 1)
        plt.plot(diameters, predictions['formula_wavelength'], 'b-', label='Formula', linewidth=2)
        plt.plot(diameters, predictions['ensemble_wavelength'], 'r--', label='Ensemble', linewidth=2)
        plt.xlabel('Diameter (nm)')
        plt.ylabel('Wavelength (nm)')
        plt.title(f'Wavelength Prediction (RI={ri})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, i + 4)
        plt.plot(diameters, predictions['formula_extinction'], 'b-', label='Formula', linewidth=2)
        plt.plot(diameters, predictions['ensemble_extinction'], 'r--', label='Ensemble', linewidth=2)
        plt.xlabel('Diameter (nm)')
        plt.ylabel('Extinction Coefficient')
        plt.title(f'Extinction Prediction (RI={ri})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Visualization saved as: prediction_comparison.png")

if __name__ == "__main__":
    print("Enhanced Nanophotonic Model - Complete Implementation")
    print("=" * 55)
    print("\nAvailable functions:")
    print("1. train_and_save_model() - Train and save the model")
    print("2. load_and_use_model() - Load model and make test predictions")
    print("3. quick_prediction_script() - Single prediction example")
    print("4. interactive_mode() - Interactive prediction interface")
    print("5. batch_prediction_from_file() - Batch predictions from CSV")
    print("6. create_sample_input_file() - Create sample input file")
    print("7. visualize_predictions() - Create prediction visualizations")
    
    # Uncomment the function you want to run:
    
    # Step 1: Train and save model (run this once)
    # train_and_save_model()
    
    # Step 2: Load and use model (run this for predictions)
    # load_and_use_model()
    
    # Step 3: Quick single prediction
    # quick_prediction_script()
    
    # Step 4: Interactive mode
    # interactive_mode()
    
    # Step 5: Create sample input and run batch predictions
    # create_sample_input_file()
    # batch_prediction_from_file('sample_input.csv', 'batch_predictions.csv')
    
    # Step 6: Create visualizations
    # visualize_predictions()
    
    print("\nTo run different functions, uncomment them in the __main__ section")