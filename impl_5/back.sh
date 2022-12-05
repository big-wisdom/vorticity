#! /usr/bin/expect

spawn scp u6047229@kingspeak.chpc.utah.edu:~/vorticity/outfield.raw ./
expect "(u6047229@kingspeak.chpc.utah.edu) Password: "
send "hicbUg-bijhux-ryjfo9\n"
interact
