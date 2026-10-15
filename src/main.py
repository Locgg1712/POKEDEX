# src/main.py

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Pokédex DSP & Machine Learning CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Train a model on local data")
    train_parser.add_argument("--data", default="data", help="Directory containing class folders")
    train_parser.add_argument("--model", default="svm", choices=["svm", "mlp"], help="Model type: svm or mlp")
    train_parser.add_argument("--epochs", type=int, default=60, help="Epochs for MLP")

    # Subcommand: predict
    predict_parser = subparsers.add_parser("predict", help="Predict Pokémon from an image")
    predict_parser.add_argument("image", help="Path to image file")

    # Subcommand: centroids
    centroids_parser = subparsers.add_parser("centroids", help="Compute feature centroids for explanation")
    centroids_parser.add_argument("--data", default="data", help="Path to dataset")

    # Subcommand: app
    subparsers.add_parser("app", help="Launch Pokédex GUI Application")

    args = parser.parse_args()

    if args.command == "train" or args.command is None:
        if args.command is None:
            # Default behavior when run without arguments
            from src.train import train
            train("data", model_type="svm")
        else:
            from src.train import train
            train(args.data, model_type=args.model, epochs=args.epochs)
    elif args.command == "predict":
        from src.predict import predict
        name, conf, expl, top3 = predict(args.image)
        print(f"\n==============================")
        print(f"🎯 Prediction: {name} ({conf*100:.1f}%)")
        print(f"Top 3 Candidates:")
        for rank, (cand, p) in enumerate(top3, 1):
            print(f"  {rank}. {cand:15s} : {p*100:.1f}%")
        print(f"==============================")
        print(f"\n{expl}")
    elif args.command == "centroids":
        from src.explain import compute_centroids
        compute_centroids(data_dir=args.data)
    elif args.command == "app":
        from src.app import PokedexApp
        app = PokedexApp()
        app.mainloop()


if __name__ == "__main__":
    main()