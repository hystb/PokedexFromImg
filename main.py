import argparse
import torch
from CNN import CNN
from train import *
from solve import *

def loadModel(model, model_load_path, device):
    model.load_state_dict(torch.load(model_load_path))
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Train or execute a CNN model for Pokedex")
    parser.add_argument('--mode', choices=['train', 'execute'], required=True, help="Choose whether to train a model or execute an existing one")
    parser.add_argument('--pathm', type=str, help="Path to the pre-trained model (required for execution mode)")
    parser.add_argument('--pathd', type=str, help="Path to the data (default value to ./PokemonData)")
    parser.add_argument('--nclasses', type=str, help="Number of differents classes the model is train for (default 150)")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    if args.mode == 'train':
        nEpoch = 100
        batchSize = 256
        learning_rate = 0.001
        model = None
        pathToData = "./PokemonData"
        if args.pathm:
            model = CNN(len(dataTrain.classes))
            model = loadModel(model, args.pathm, device)
            model.train()
            print("Model Loaded")
        if args.pathd:
            pathToData = args.pathd
        train(model=model, learningRate=learning_rate, nEpoch=nEpoch, device=device, batchSize=batchSize, pathToData=pathToData)
    elif args.mode == 'execute':
        if args.nclasses and args.pathm and args.pathd:
            solve(args.nclasses, args.pathm, device, args.pathd)
        else:
            print("Missing --pathm, --pathd and/or --nclasses use --help for more informations")
            
        
if __name__ == "__main__":
    main()
