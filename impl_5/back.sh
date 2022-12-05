#! /usr/bin/expect

spawn scp u6047229@kingspeak.chpc.utah.edu:~/vorticity/parallel_shared_memory_gpu.cu ./
expect "(u6047229@kingspeak.chpc.utah.edu) Password: "
send "hicbUg-bijhux-ryjfo9\n"
interact
