readvdnames = lambda x: open(x).read().rstrip().split('\n')

fail_uids = readvdnames("fail_all.txt")

import os

for unique_id in fail_uids:
    d, _ = unique_id.split('-')
    fn = "{}/{}.npy".format(d, unique_id)
    fn2 = "{}/{}_lung_mask.npy".format(d, unique_id)

    cmd1 = "/bin/rm -rf {}".format(fn)
    cmd2 = "/bin/rm -rf {}".format(fn2)
    print (cmd1)
    print (cmd2)
    os.system(cmd1)
    os.system(cmd2)



