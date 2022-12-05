#! /usr/bin/expect 

spawn scp ./gpu.h u6047229@kingspeak.chpc.utah.edu:~/vorticity
expect "(u6047229@kingspeak.chpc.utah.edu) Password: "
send "hicbUg-bijhux-ryjfo9\n"
interact

