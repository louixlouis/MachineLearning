import configargparse
import os

def args_parser():
    parser = configargparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', is_config_file=True, help='Config file path')

    # Dataets.
    parser.add('data_type', type=str, default='blender', help='[blender, llff, custom]')
    parser.add('data_name', type=str)
    parser.add('data_root', type=str)
    parser.add('downsample', type=int, default=0)

    # For blender
    parser.set_defaults(bg_white=False)
    parser.add_argument('--bg_white', dest='bg_white', action='store_true')

    # 
    parser.add('--L_x', type=int, default=10)
    parser.add('--L_d', type=int, default=4)
    parser.add('--netDepth', type=int, default=8)
    parser.add('--netWidth', type=int, default=256)

    # Training.
    parser.add('--lr', type=float, default=5e-4)
    parser.add('')
    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = args_parser()
    print(opts)