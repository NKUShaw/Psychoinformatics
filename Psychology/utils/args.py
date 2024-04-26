import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train the Networks on images and target masks')
    parser.add_argument('--model', type=str, default='VGG16', help='model name')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=120, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help="type of optimizer")
    parser.add_argument('--lr', '-l', metavar='LR', type=float, default=5e-4, help='Learning rate', dest='lr')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--target', type=str, default='atypical', help="target")
    parser.add_argument('--loss', type=str, default='mseloss', help="loss")
    parser.add_argument('--rs', type=float, default=1, help="regularization")
    parser.add_argument('--es', type=int, default=10, help="Early Stopping")
    return parser.parse_args()