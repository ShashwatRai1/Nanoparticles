import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, LSTM, GRU
from keras.optimizers import Adam, AdamW
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import os, seaborn as sns, pickle, joblib, json

print(f"TensorFlow: {tf.__version__}, Keras: {keras.__version__}")
np.random.seed(42); tf.random.set_seed(42)

class InverseNanophotonicModel:
    def __init__(self, data_path):
        self.data_path, self.model_dir = data_path, "inverse_models"; os.makedirs(self.model_dir, exist_ok=True)
        self.wavelength_ranges, self.neural_models, self.scalers, self.ensemble_models = self._define_wavelength_ranges(), {}, {}, {}
    
    def _define_wavelength_ranges(self):
        return {(300, 500): 'UV-Blue', (501, 600): 'Green', (601, 700): 'Red', (701, 900): 'NIR-1', (901, 1200): 'NIR-2'}
    
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
            print("Cleaning NaN/infinite values..."); combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna(); print(f"After cleaning: {len(combined_data)} data points")
        return combined_data
    
    def find_wavelength_range(self, wavelength):
        for (min_w, max_w) in self.wavelength_ranges.keys():
            if min_w <= wavelength <= max_w: return (min_w, max_w)
        closest_range = min(self.wavelength_ranges.keys(), key=lambda x: min(abs(wavelength - x[0]), abs(wavelength - x[1])))
        print(f"Warning: Wavelength {wavelength} outside ranges. Using {closest_range}.")
        return closest_range
    
    def calculate_inverse_features(self, df):
        wavelength, extinction = df['resonance_wavelength'], df['extinction']
        features = pd.DataFrame()
        features['wavelength'], features['extinction'] = wavelength, extinction
        features['wavelength_squared'], features['wavelength_cubed'] = wavelength ** 2, wavelength ** 3
        features['extinction_squared'], features['extinction_cubed'] = extinction ** 2, extinction ** 3
        features['w_times_e'], features['w_squared_times_e'] = wavelength * extinction, wavelength ** 2 * extinction
        features['w_times_e_squared'], features['w_cubed_times_e'] = wavelength * extinction ** 2, wavelength ** 3 * extinction
        features['w_squared_times_e_squared'], features['w_times_e_cubed'] = wavelength ** 2 * extinction ** 2, wavelength * extinction ** 3
        features['log_wavelength'], features['log_extinction'] = np.log(wavelength + 1), np.log(np.abs(extinction) + 1)
        features['sqrt_wavelength'], features['sqrt_extinction'] = np.sqrt(wavelength), np.sqrt(np.abs(extinction))
        features['inv_wavelength'], features['inv_extinction'] = 1 / (wavelength + 1e-6), 1 / (np.abs(extinction) + 1e-6)
        features['exp_w_e'] = np.exp(-wavelength * np.abs(extinction) / 10000)
        features['tanh_we'] = np.tanh(wavelength * np.abs(extinction) / 1000)
        features['sin_wavelength'], features['cos_wavelength'] = np.sin(wavelength / 100), np.cos(wavelength / 100)
        features['sin_extinction'], features['cos_extinction'] = np.sin(np.abs(extinction) / 100), np.cos(np.abs(extinction) / 100)
        return features
    
    def train_ensemble_models(self, df):
        print("Training inverse ensemble models...")
        for range_key in self.wavelength_ranges.keys():
            print(f"Training models for wavelength range {range_key}")
            mask = (df['resonance_wavelength'] >= range_key[0]) & (df['resonance_wavelength'] <= range_key[1])
            range_data = df[mask].copy()
            if len(range_data) < 10: print(f"Insufficient data for range {range_key} ({len(range_data)} samples)"); continue
            X = self.calculate_inverse_features(range_data[['resonance_wavelength', 'extinction']])
            y = range_data[['diameter', 'refractive_index']].values
            mask_valid = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
            X, y = X[mask_valid], y[mask_valid]
            if len(X) < 10: print(f"Insufficient valid data for range {range_key}"); continue
            rf_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
            gb_model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
            rf_model.fit(X, y); gb_model.fit(X, y)
            self.ensemble_models[range_key] = {'rf': rf_model, 'gb': gb_model}
            print(f"Range {range_key}: Trained with {len(X)} samples")
    
    def create_inverse_neural_model(self, input_dim):
        model = Sequential([Dense(256, activation='relu', input_shape=(input_dim,)), BatchNormalization(), Dropout(0.3), Dense(128, activation='relu'), BatchNormalization(), Dropout(0.2), Dense(64, activation='relu'), Dropout(0.1), Dense(32, activation='relu'), Dense(2, activation='linear')])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_neural_models(self, df):
        print("Training inverse neural network models...")
        for range_key in self.wavelength_ranges.keys():
            print(f"Training neural model for wavelength range {range_key}")
            mask = (df['resonance_wavelength'] >= range_key[0]) & (df['resonance_wavelength'] <= range_key[1])
            range_data = df[mask].copy()
            if len(range_data) < 20: print(f"Insufficient data for neural model in range {range_key}"); continue
            X = self.calculate_inverse_features(range_data[['resonance_wavelength', 'extinction']])
            y = range_data[['diameter', 'refractive_index']].values
            mask_valid = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
            X, y = X[mask_valid], y[mask_valid]
            if len(X) < 20: print(f"Insufficient valid data for neural model in range {range_key}"); continue
            X_scaler, y_scaler = StandardScaler(), StandardScaler()
            X_scaled, y_scaled = X_scaler.fit_transform(X), y_scaler.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
            model = self.create_inverse_neural_model(X_scaled.shape[1])
            early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, callbacks=[early_stopping, reduce_lr], verbose=0)
            self.neural_models[range_key] = model; self.scalers[range_key] = {'X_scaler': X_scaler, 'y_scaler': y_scaler}
            train_loss = model.evaluate(X_train, y_train, verbose=0)[0]; val_loss = model.evaluate(X_test, y_test, verbose=0)[0]
            print(f"Range {range_key}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def predict_inverse(self, wavelengths, extinctions):
        if not isinstance(wavelengths, list): wavelengths = [wavelengths]
        if not isinstance(extinctions, list): extinctions = [extinctions] * len(wavelengths)
        results = {'neural_diameter': [], 'neural_refractive_index': [], 'rf_diameter': [], 'rf_refractive_index': [], 'gb_diameter': [], 'gb_refractive_index': [], 'ensemble_diameter': [], 'ensemble_refractive_index': []}
        for w, e in zip(wavelengths, extinctions):
            range_key = self.find_wavelength_range(w)
            df = pd.DataFrame({'resonance_wavelength': [w], 'extinction': [e]})
            features = self.calculate_inverse_features(df)
            neural_d, neural_ri, rf_d, rf_ri, gb_d, gb_ri = None, None, None, None, None, None
            if range_key in self.neural_models and range_key in self.scalers:
                X_scaler, y_scaler, model = self.scalers[range_key]['X_scaler'], self.scalers[range_key]['y_scaler'], self.neural_models[range_key]
                X_scaled = X_scaler.transform(features.values); y_pred_scaled = model.predict(X_scaled, verbose=0); y_pred = y_scaler.inverse_transform(y_pred_scaled)
                neural_d, neural_ri = y_pred[0]
            if range_key in self.ensemble_models:
                rf_pred = self.ensemble_models[range_key]['rf'].predict(features.values); gb_pred = self.ensemble_models[range_key]['gb'].predict(features.values)
                rf_d, rf_ri = rf_pred[0]; gb_d, gb_ri = gb_pred[0]
            if all(x is not None for x in [neural_d, rf_d, gb_d]):
                ensemble_d = 0.5 * neural_d + 0.3 * rf_d + 0.2 * gb_d; ensemble_ri = 0.5 * neural_ri + 0.3 * rf_ri + 0.2 * gb_ri
            else: ensemble_d, ensemble_ri = neural_d or rf_d or gb_d, neural_ri or rf_ri or gb_ri
            results['neural_diameter'].append(neural_d); results['neural_refractive_index'].append(neural_ri)
            results['rf_diameter'].append(rf_d); results['rf_refractive_index'].append(rf_ri)
            results['gb_diameter'].append(gb_d); results['gb_refractive_index'].append(gb_ri)
            results['ensemble_diameter'].append(ensemble_d); results['ensemble_refractive_index'].append(ensemble_ri)
        return pd.DataFrame(results)
    
    def save_inverse_model(self, filename="inverse_nanophotonic_model"):
        save_path = os.path.join(self.model_dir, filename); os.makedirs(save_path, exist_ok=True)
        neural_models_path = os.path.join(save_path, "neural_models"); os.makedirs(neural_models_path, exist_ok=True)
        for range_key, model in self.neural_models.items():
            model_name = f"neural_{range_key[0]}_{range_key[1]}.keras"; model.save(os.path.join(neural_models_path, model_name))
        ensemble_models_path = os.path.join(save_path, "ensemble_models"); os.makedirs(ensemble_models_path, exist_ok=True)
        for range_key, models in self.ensemble_models.items():
            range_name = f"{range_key[0]}_{range_key[1]}"; joblib.dump(models['rf'], os.path.join(ensemble_models_path, f"rf_{range_name}.pkl")); joblib.dump(models['gb'], os.path.join(ensemble_models_path, f"gb_{range_name}.pkl"))
        scalers_path = os.path.join(save_path, "scalers"); os.makedirs(scalers_path, exist_ok=True)
        for range_key, scalers in self.scalers.items():
            range_name = f"{range_key[0]}_{range_key[1]}"; joblib.dump(scalers['X_scaler'], os.path.join(scalers_path, f"X_scaler_{range_name}.pkl")); joblib.dump(scalers['y_scaler'], os.path.join(scalers_path, f"y_scaler_{range_name}.pkl"))
        metadata = {'wavelength_ranges': [list(key) for key in self.wavelength_ranges.keys()], 'model_info': {'neural_models': list(self.neural_models.keys()), 'ensemble_models': list(self.ensemble_models.keys()), 'scalers': list(self.scalers.keys())}}
        with open(os.path.join(save_path, "metadata.json"), 'w') as f: json.dump(metadata, f, indent=2, default=str)
        print(f"Inverse model saved to: {save_path}"); return save_path
    
    def load_inverse_model(self, filename="inverse_nanophotonic_model"):
        load_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(load_path): raise FileNotFoundError(f"Model directory not found: {load_path}")
        with open(os.path.join(load_path, "metadata.json"), 'r') as f: metadata = json.load(f)
        neural_models_path = os.path.join(load_path, "neural_models"); self.neural_models = {}
        for range_key_list in metadata['wavelength_ranges']:
            range_key = tuple(range_key_list); model_name = f"neural_{range_key[0]}_{range_key[1]}.keras"; model_path = os.path.join(neural_models_path, model_name)
            if os.path.exists(model_path): self.neural_models[range_key] = load_model(model_path)
        ensemble_models_path = os.path.join(load_path, "ensemble_models"); self.ensemble_models = {}
        for range_key_list in metadata['wavelength_ranges']:
            range_key = tuple(range_key_list); range_name = f"{range_key[0]}_{range_key[1]}"; rf_path = os.path.join(ensemble_models_path, f"rf_{range_name}.pkl"); gb_path = os.path.join(ensemble_models_path, f"gb_{range_name}.pkl")
            if os.path.exists(rf_path) and os.path.exists(gb_path): self.ensemble_models[range_key] = {'rf': joblib.load(rf_path), 'gb': joblib.load(gb_path)}
        scalers_path = os.path.join(load_path, "scalers"); self.scalers = {}
        for range_key_list in metadata['wavelength_ranges']:
            range_key = tuple(range_key_list); range_name = f"{range_key[0]}_{range_key[1]}"; x_scaler_path = os.path.join(scalers_path, f"X_scaler_{range_name}.pkl"); y_scaler_path = os.path.join(scalers_path, f"y_scaler_{range_name}.pkl")
            if os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path): self.scalers[range_key] = {'X_scaler': joblib.load(x_scaler_path), 'y_scaler': joblib.load(y_scaler_path)}
        print(f"Inverse model loaded from: {load_path}"); print(f"Loaded {len(self.neural_models)} neural models, {len(self.ensemble_models)} ensemble models, {len(self.scalers)} scaler sets")
    
    def interactive_inverse_prediction(self):
        print("\nInverse Design Prediction Interface"); print("=" * 40)
        while True:
            try:
                wavelength = float(input("\nWavelength (nm) [-1 to exit]: "))
                if wavelength == -1: break
                extinction = float(input("Extinction coefficient: "))
                predictions = self.predict_inverse(wavelength, extinction)
                print(f"\nInverse Design Results for λ={wavelength}nm, E={extinction}:")
                if predictions['neural_diameter'][0]: print(f"Neural:   D={predictions['neural_diameter'][0]:.2f}nm, RI={predictions['neural_refractive_index'][0]:.3f}")
                if predictions['ensemble_diameter'][0]: print(f"Ensemble: D={predictions['ensemble_diameter'][0]:.2f}nm, RI={predictions['ensemble_refractive_index'][0]:.3f}")
            except ValueError: print("Invalid input. Enter numeric values.")
            except Exception as e: print(f"Error: {e}")
    
    def evaluate_inverse_performance(self, df):
        print("\nEvaluating inverse model performance...")
        for range_key in self.wavelength_ranges.keys():
            mask = (df['resonance_wavelength'] >= range_key[0]) & (df['resonance_wavelength'] <= range_key[1])
            range_data = df[mask].copy()
            if len(range_data) < 10: continue
            wavelengths = range_data['resonance_wavelength'].tolist(); extinctions = range_data['extinction'].tolist()
            actual_diameters = range_data['diameter'].values; actual_ris = range_data['refractive_index'].values
            predictions = self.predict_inverse(wavelengths, extinctions)
            if predictions['ensemble_diameter'][0] is not None:
                ensemble_diameters = [x for x in predictions['ensemble_diameter'] if x is not None]; ensemble_ris = [x for x in predictions['ensemble_refractive_index'] if x is not None]
                if len(ensemble_diameters) > 0:
                    mae_d = mean_absolute_error(actual_diameters, ensemble_diameters); rmse_d = np.sqrt(mean_squared_error(actual_diameters, ensemble_diameters)); r2_d = r2_score(actual_diameters, ensemble_diameters)
                    mae_ri = mean_absolute_error(actual_ris, ensemble_ris); rmse_ri = np.sqrt(mean_squared_error(actual_ris, ensemble_ris)); r2_ri = r2_score(actual_ris, ensemble_ris)
                    print(f"Range {range_key} - Diameter: MAE: {mae_d:.2f}nm, RMSE: {rmse_d:.2f}nm, R²: {r2_d:.3f}")
                    print(f"Range {range_key} - RI: MAE: {mae_ri:.3f}, RMSE: {rmse_ri:.3f}, R²: {r2_ri:.3f}")

def train_and_save_inverse_model():
    print("Training and saving inverse model...")
    model = InverseNanophotonicModel("resonancedata1.xlsx")
    df = model.load_data(); model.train_ensemble_models(df); model.train_neural_models(df); model.evaluate_inverse_performance(df)
    save_path = model.save_inverse_model("my_inverse_model_v1"); print(f"\nInverse model training complete and saved to: {save_path}")
    return model

def load_and_use_inverse_model():
    print("Loading pre-trained inverse model...")
    model = InverseNanophotonicModel(data_path=None); model.load_inverse_model("my_inverse_model_v1")
    test_wavelengths = [400, 500, 600, 700, 800]; test_extinctions = [1000, 2000, 3000, 4000, 5000]
    predictions = model.predict_inverse(test_wavelengths, test_extinctions)
    print("\nInverse Prediction Results:"); print(predictions[['neural_diameter', 'neural_refractive_index', 'ensemble_diameter', 'ensemble_refractive_index']])
    return model

def quick_inverse_prediction():
    model = InverseNanophotonicModel(data_path=None); model.load_inverse_model("my_inverse_model_v1")
    wavelength, extinction = 550, 2500
    prediction = model.predict_inverse(wavelength, extinction)
    print(f"\nInverse Prediction for λ={wavelength}nm, E={extinction}:"); print(f"Diameter: {prediction['ensemble_diameter'][0]:.2f} nm"); print(f"Refractive Index: {prediction['ensemble_refractive_index'][0]:.3f}")

def interactive_inverse_mode():
    print("Starting inverse interactive mode...")
    model = InverseNanophotonicModel(data_path=None)
    try: model.load_inverse_model("my_inverse_model_v1"); model.interactive_inverse_prediction()
    except FileNotFoundError: print("No trained inverse model found. Please run train_and_save_inverse_model() first.")

def batch_inverse_prediction(input_file, output_file):
    model = InverseNanophotonicModel(data_path=None); model.load_inverse_model("my_inverse_model_v1")
    input_data = pd.read_csv(input_file); required_columns = ['wavelength', 'extinction']
    if not all(col in input_data.columns for col in required_columns): raise ValueError(f"Input file must contain columns: {required_columns}")
    wavelengths = input_data['wavelength'].tolist(); extinctions = input_data['extinction'].tolist()
    predictions = model.predict_inverse(wavelengths, extinctions)
    result_data = pd.concat([input_data, predictions], axis=1); result_data.to_csv(output_file, index=False)
    print(f"Batch inverse predictions saved to: {output_file}"); return result_data

def create_inverse_sample_input():
    sample_data = pd.DataFrame({'wavelength': [400, 450, 500, 550, 600, 650, 700, 750, 800, 850], 'extinction': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]})
    sample_data.to_csv('inverse_sample_input.csv', index=False); print("Inverse sample input file created: inverse_sample_input.csv"); return sample_data

def visualize_inverse_predictions():
    model = InverseNanophotonicModel(data_path=None); model.load_inverse_model("my_inverse_model_v1")
    wavelengths = np.linspace(350, 850, 50); extinctions = [1000, 2500, 5000]
    plt.figure(figsize=(15, 8))
    for i, ext in enumerate(extinctions):
        predictions = model.predict_inverse(wavelengths.tolist(), [ext] * len(wavelengths))
        plt.subplot(2, 3, i + 1); plt.plot(wavelengths, predictions['ensemble_diameter'], 'r-', linewidth=2, label=f'Extinction={ext}')
        plt.xlabel('Wavelength (nm)'); plt.ylabel('Predicted Diameter (nm)'); plt.title(f'Diameter vs Wavelength (E={ext})'); plt.grid(True, alpha=0.3)
        plt.subplot(2, 3, i + 4); plt.plot(wavelengths, predictions['ensemble_refractive_index'], 'b-', linewidth=2, label=f'Extinction={ext}')
        plt.xlabel('Wavelength (nm)'); plt.ylabel('Predicted Refractive Index'); plt.title(f'RI vs Wavelength (E={ext})'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('inverse_prediction_plots.png', dpi=300, bbox_inches='tight'); plt.show()
    print("Inverse visualization saved as: inverse_prediction_plots.png")

if __name__ == "__main__":
    interactive_inverse_mode()
    print("Inverse Nanophotonic Design Model"); print("=" * 40)
    print("\nAvailable functions:")
    print("1. train_and_save_inverse_model() - Train and save the inverse model")
    print("2. load_and_use_inverse_model() - Load model and make test predictions")
    print("3. quick_inverse_prediction() - Single inverse prediction example")
    print("4. interactive_inverse_mode() - Interactive inverse prediction interface")
    print("5. batch_inverse_prediction() - Batch inverse predictions from CSV")
    print("6. create_inverse_sample_input() - Create sample input file")
    print("7. visualize_inverse_predictions() - Create inverse prediction visualizations")
    interactive_inverse_mode()