
import matplotlib.pyplot as plt
import numpy as np

def calculate_probs(n=10):
    weights = [n - i for i in range(n)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return probs

def show_stats(n=10):
    probs = calculate_probs(n)
    print(f"Population Size: {n}")
    print(f"{'Rank':<5} {'Prob':<10} {'Cumul':<10}")
    print("-" * 30)
    cumul = 0
    for i, p in enumerate(probs):
        cumul += p
        print(f"{i+1:<5} {p:.1%}       {cumul:.1%}")

    print("\nInsights:")
    print(f"- Top 1 (Rank 1) chance: {probs[0]:.1%}")
    print(f"- Top 3 (Rank 1-3) chance: {sum(probs[:3]):.1%}")
    print(f"- Top 5 (Rank 1-5) chance: {sum(probs[:5]):.1%}")

if __name__ == "__main__":
    show_stats(10)
