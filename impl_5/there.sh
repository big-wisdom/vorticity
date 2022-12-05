#! /usr/bin/expect 

spawn scp ./parallel_shared_memory_gpu.cu u6047229@kingspeak.chpc.utah.edu:~/vorticity
expect "(u6047229@kingspeak.chpc.utah.edu) Password: "
send "hicbUg-bijhux-ryjfo9\n"
interact

spawn scp ./distributed_memory_cpu.c u6047229@kingspeak.chpc.utah.edu:~/vorticity
expect "(u6047229@kingspeak.chpc.utah.edu) Password: "
send "hicbUg-bijhux-ryjfo9\n"
interact

spawn scp ./gpu.h u6047229@kingspeak.chpc.utah.edu:~/vorticity
expect "(u6047229@kingspeak.chpc.utah.edu) Password: "
send "hicbUg-bijhux-ryjfo9\n"
interact
