import argparse

# %% Input parameters
def load_user_input():
    '''
    Function that stores the user input
    Returns:
        parser.parse_args(): it passes the parser state

    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--b_size', type=int, default=128, help='Batch size: Table 4 of the original paper')
    parser.add_argument('--context_size', type=int, default=64, help='Context size: m')
    parser.add_argument('--input_size', type=int, default=64, help='Input size: n')
    parser.add_argument('--qk_size', type=int, default=16, help='Key size: k')
    parser.add_argument('--heads', type=int, default=4, help='Number of heads: h')
    parser.add_argument('--n_rows_plot', type=int, default=8, help='Number of rows to include in the plot of CIFAR-10')
    parser.add_argument('--n_col_plot', type=int, default=8, help='Number of columns to include in the plot of CIFAR-10')
    parser.add_argument('--epochs', type=int, default=90,
                        help='Number of epochs: suggested by Robert-Jan Bruintjes and the LambdaNetworks paper')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the Adam')
    parser.add_argument('--initial_lr', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--th', type=int, default=5, help='Threshold number of epochs to change scheduler')
    parser.add_argument('--model_type', type=int, default=1, help='Type of model: 0 = baseline; 1 = lambda')
    parser.add_argument('--resume', type=bool, default=False, help='Resume from the latest checkpoint')
    parser.add_argument('--smoothing', type=bool, default=True,
                        help='Switch which defines whether label smoothing should take place')
    parser.add_argument('--BN_gamma', type=bool, default=True,
                        help='Initialisation value of the gamma parameter of the last BN layer')
    parser.add_argument('--cp_dir', type=str, default=".\\Checkpoints", help='Base checkpoint folder')

    return parser.parse_args()
