import os

from src.data import load_weights
from src.models import fit_all_models
from src.chart import generate_html

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "weights.csv")
    output_path = os.path.join(os.path.dirname(__file__), "..", "docs", "index.html")

    df = load_weights(csv_path)
    models = fit_all_models(df["days"].values, df["weight"].values)
    generate_html(df, models, output_path)

    print(f"Generated {output_path}")
    print(f"  Data points: {len(df)}")
    print(f"  Models fitted: {len(models)}")
    for name, model in models.items():
        print(f"    {name}: R^2 = {model['r_squared']:.4f}")

if __name__ == "__main__":
    main()
