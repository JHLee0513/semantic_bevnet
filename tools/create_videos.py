import argparse
import os
import subprocess

def process_dir(in_dir, out_dir, seq=None, tag=None, hd=False):

    if seq is None:
        sequences = [s for s in os.listdir(in_dir) 
                     if os.path.isdir(in_dir)]
    else:
        sequences = [seq]


    sequences = sorted(sequences)
    output_videos = []
    for seq in sequences:
        if not os.path.isdir(os.path.join(in_dir, seq)):
            continue

        input_format = os.path.join(in_dir, seq, '%05d.png')
        out_name = seq if tag is None else '{}_{}'.format(tag, seq)
        output_file = os.path.join(out_dir, '{}.mp4'.format(out_name))
        print('{} --> {}'.format(input_format, output_file))

        if hd:
            hd_parms = ['-vb', '20M']
        else:
            hd_parms = []

        c = subprocess.call(['ffmpeg', '-r', '25', '-i', input_format] + hd_parms +
                             ['-c:v', 'libopenh264', '-vf', 'fps=25', '-pix_fmt', 'yuv420p',
                             output_file])
        if c != 0:
            print('Failed to create {}!'.format(out_name))

        output_videos.append(output_file)

    return output_videos, sequences

def compare(videos, name, tag, output_dir, hd=False):

    cmd = ['ffmpeg']

    for v in videos:
        cmd.append('-i')
        cmd.append(v)

    if hd:
        cmd.append('-vb')
        cmd.append('20M')
    cmd.append('-filter_complex')
    cmd.append('vstack')
    output = os.path.join(output_dir, '{}_{}.mp4'.format(tag, name))
    cmd.append(output)

    print(' '.join(cmd))
    c = subprocess.call(cmd)

    if c != 0:
        print('Failed to merge {}!'.format(name))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert prediction images to video.')
    parser.add_argument('-i','--input_dir', nargs='+',
            help='Path to the directory that contains the prediction sequences.',
            required=True)
    parser.add_argument('-t', '--tag', nargs='+', required=False, default=None,
            help='Optinally tag the output videos.')
    parser.add_argument('-s', '--sequence', type=str, default=None,
            help='Only make video of the specified sequence.')

    parser.add_argument('-o', '--output_dir', type=str, default='.',
            help='Path to the output directory.')
    parser.add_argument('-hd', action='store_true', help='Increase video quality.')

    args = parser.parse_args()


    tags = []
    output_videos = []
    pre_s = None
    similar_sequences = True
    for i in range(len(args.input_dir)):
        tag = None
        if args.tag:
            tag = args.tag[i]

        if len(args.input_dir) > 1 and tag is None:
            tag = 'n{}'.format(i)

        tags.append(tag)
        o, s = process_dir(args.input_dir[i], args.output_dir, seq=args.sequence, tag=tag, hd=args.hd)
        output_videos.append(o)
        if pre_s is not None:
            similar_sequences = similar_sequences and all([x==y for x,y in zip(s, pre_s)])
            pre_s = s


    ## Comparision
    if len(output_videos) > 1 and similar_sequences:
        for i, comp in enumerate(zip(*output_videos)):
            compare(comp, s[i], '_'.join(tags), args.output_dir, hd=args.hd)
