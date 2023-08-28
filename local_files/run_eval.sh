#cd ../
# dist
#./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./path/to/ckpts.pth 8
#./tools/dist_test.sh ./ckpts/bevformer_tiny.py ./ckpts/bevformer_tiny_epoch_24.pth 1

# single gpu
# 目前版本不支持直接采用test.py的运行
# 可以将这里注释了，用来进行调试，测出来的结果好像也是正常的
#    if not distributed:
#         assert False TODO
#python tools/test.py ./ckpts/bevformer_tiny.py ./ckpts/bevformer_tiny_epoch_24.pth --eval bbox

#--show
#--show-dir
#/home/dell/下载/debug
# 目前不支持show的模式，需要修改def show_results(self, data, result, out_dir)的实现方式