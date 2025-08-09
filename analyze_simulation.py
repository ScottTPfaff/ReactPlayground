"""Analyze simulation data and output summary statistics."""
import json
from statistics import mean


def main() -> None:
    with open('simulation_data.json') as f:
        data = json.load(f)
    field_avg = mean(data['field'])
    dominant_freqs = data.get('dominant_freqs', [])[:5]
    print(f"Average field: {field_avg:.4f}")
    print("First 5 dominant frequencies:", dominant_freqs)


if __name__ == "__main__":
    main()
