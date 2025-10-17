#!/home/lin/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/root/miniconda3/envs/aloha/bin/python
#!/home/lin/miniconda3/envs/aloha/bin/python
"""

import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetDir', action='store', type=str, help='datasetDir',
                        default="/home/agilex/data", required=False)
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    for f in os.listdir(args.datasetDir):
        if f.startswith("episode") and not f.endswith(".tar.gz"):
            os.system(f"cp {args.datasetDir}/episode1/instructions.json {args.datasetDir}/{f}/")
    print("Done")


if __name__ == '__main__':
    main()
