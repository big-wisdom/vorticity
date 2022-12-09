% axis
datasize = [1,2,4,8];
numblocks = [600,300,150,75];
numcores = [1,2,4,8];


% 1,2,3
parasharedCPU = [0.030272,	0.030336,	0.03028,	0.030302;
    0.060558,	0.060577,	0.060516,	0.060557;
    0.120475,	0.120473,	0.12048,	0.12053;
    0.240302,	0.24018,	0.239777,	0.24028];
parasharedGPU = [0.001566,	0.001156,	0.001136,	0.001396;
    0.003064,	0.002242,	0.00221,	0.00271;
    0.007382,	0.005988,	0.005974,	0.007313;
    0.015991,	0.013425,	0.013492,	0.016551];

hold on
for i = 1:4
    plot(datasize, parasharedCPU(:,i), '-*')
end
title("Time spent vs input size on different numbers of cores for shared memory CPU")
xlabel("Datasize (multiples of 3120000)")
ylabel("Time (seconds)")
legend("1 core", "2 cores", "4 cores", "8 cores")
hold off

figure
hold on
for i = 1:4
    plot(numcores, parasharedCPU(i,:),'-o')
end
title("Time spent vs number of threads for different input sizes for shared memory CPU")
xlabel("Number of threads")
ylabel("Time (seconds)")
legend("1x input", "2x input", "4x input", "8x input")
hold off

figure
hold on
for i = 1:4
    plot(datasize, parasharedGPU(:,i), '-*')
end
title("Time spent vs input size on different numbers of blocks for shared memory GPU")
xlabel("Datasize (multiples of 3120000)")
ylabel("Time (seconds)")
legend("600 blocks", "300 blocks", "150 blocks", "75 blocks")
hold off

figure
hold on
for i = 1:4
    plot(numblocks, parasharedGPU(i,:),'-o')
end
title("Time spent vs number of blocks for different input sizes for shared memory GPU")
xlabel("Number of blocks")
ylabel("Time (seconds)")
legend("1x input", "2x input", "4x input", "8x input")
hold off

% 4,5
datasize = [1,2,3,4];
distCPUv = [0.014762, 0.015037, 0.014744, 0.015153];
% 2x size
distCPUc = [0.028423, 0.014705, 0.008204, 0.005713];
distGPUv = [0.129209, 0.394031];
distGPUc = [0.125691, 0.394031];

hold on
figure
plot(datasize, distCPUv)
plot([1,2], distGPUv)
title("Time spent on core 0 vs data size and core count")
xlabel("Datasize (multiples of 3120000)")
ylabel("Time (seconds)")
legend("Distributed Mem CPU", "Distributed Mem GPU")
hold off

hold on
figure
plot(numcores, distCPUc)
plot([1,2], distGPUc)
title("Time spent on core 0 vs core count for fixed data size")
ylabel("Time (in seconds)")
xlabel("Number of cores")
legend("Distributed Mem CPU", "Distributed Mem GPU")
hold off

hold on
figure
semilogy(numcores, distCPUc)
title("Time spent on core 0 vs core count for fixed data size")
ylabel("Time (in seconds)")
xlabel("Number of cores")
legend("Distributed Mem CPU", "Distributed Mem GPU")
hold off