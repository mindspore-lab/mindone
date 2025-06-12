export DEVICE_ID=2

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ 8`
gap=`expr $avg \- 1`
start=`expr $DEVICE_ID \* $avg`
end=`expr $start \+ $gap`
cmdopt=$start"-"$end
echo "start set cpu $cmdopt"

taskset -c $cmdopt python qwen25_generate.py &> test_page_attention.log &
