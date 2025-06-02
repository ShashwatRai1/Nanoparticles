import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load and combine data
def load_data(path):
    sheet_to_ri = {'1':1.0,'2':1.2,'3':1.3,'4':1.33,'5':1.4,'6':1.5}
    frames = []
    for sheet, ri in sheet_to_ri.items():
        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = df.columns.str.strip().str.lower()
        mapping = {
            'size (diameter)': 'diameter',
            'resonance wavelength': 'resonance_wavelength',
            'resonance intensity (extinction)': 'extinction'
        }
        df = df.rename(columns={k:v for k,v in mapping.items() if k in df.columns})
        df['refractive_index'] = ri
        frames.append(df[['diameter','refractive_index','resonance_wavelength','extinction']])
    return pd.concat(frames, ignore_index=True).dropna()

# Print precise formulas with R² per diameter range
def print_precise_formulas(df, degree=2):
    ranges = [(0,50),(51,100),(101,150),(151,200),(201,250),(251,300)]

    for dmin, dmax in ranges:
        subset = df[(df['diameter'] >= dmin) & (df['diameter'] <= dmax)]
        n = len(subset)
        if n < 10:
            print(f"Skipping range {dmin}-{dmax} nm: only {n} samples")
            continue

        # Prepare features
        X = subset[['diameter','refractive_index']].values
        y_wl = subset['resonance_wavelength'].values
        y_ext = subset['extinction'].values

        # Fit polynomial transformer for this range
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xp = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(['diameter','refractive_index'])

        # Fit models
        lr_wl  = LinearRegression().fit(Xp, y_wl)
        lr_ext = LinearRegression().fit(Xp, y_ext)

        # Compute R²
        r2_wl  = r2_score(y_wl, lr_wl.predict(Xp))
        r2_ext = r2_score(y_ext, lr_ext.predict(Xp))

        # Build formula strings
        coef_wl = lr_wl.coef_
        int_wl  = lr_wl.intercept_
        terms_wl = " + ".join([f"({coef_wl[i]:.6f}*{feature_names[i]})" for i in range(len(coef_wl))])

        coef_ext = lr_ext.coef_
        int_ext  = lr_ext.intercept_
        terms_ext = " + ".join([f"({coef_ext[i]:.6f}*{feature_names[i]})" for i in range(len(coef_ext))])

        # Print results
        print(f"\nRange {dmin}-{dmax} nm (n={n})")
        print(f"  λᵣ(D,n) ≈ {int_wl:.6f} + {terms_wl}")
        print(f"    R² (λᵣ) = {r2_wl:.4f}")
        print(f"  ε(D,n)  ≈ {int_ext:.6f} + {terms_ext}")
        print(f"    R² (ε)  = {r2_ext:.4f}")

if __name__ == '__main__':
    data_path = 'resonancedata.xlsx'  # update this path if needed
    df_all = load_data(data_path)
    print_precise_formulas(df_all)
