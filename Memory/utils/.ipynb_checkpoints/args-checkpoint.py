import argparse
#python test.py --model=VGG16 --load=./saved_weights/best_model_0.56.pth
def get_args():
    parser = argparse.ArgumentParser(description='Train the Networks on images and target masks')
    parser.add_argument('--model', type=str, default='CNN', help='model name')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--lr', '-l', metavar='LR', type=float, default=5e-4, help='Learning rate', dest='lr')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay for SGD')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    return parser.parse_args()