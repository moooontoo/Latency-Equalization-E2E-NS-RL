# Latency-Equalization-E2E-NS-RL

[ Latency Equalization Policy of End-to-End Network Slicing Based on Reinforcement Learning | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9903906)  
This repository contains Python scripts implementing the slicing resource allocation algorithms (RAN and CN) and latency equalization policies proposed in the paper "Latency Equalization Policy of End-to-End Network Slicing Based on Reinforcement Learning". 
Run "DSDP.py"/"DTDP.py"/"static.py" to excute the latency equalization policies to generate the results.
The slicing algorithms in RAN and CN have been abstracted into functions (rlRAN, test) in "ran.py" and "Test.py". 
"config.py" contains some parameter information for the wireless network. The "user_plus" folder contains the updated rlRAN() , which only adjust the number of users.
The "data" folder contains "virtualnetworkTP.txt", which records the information of the SFC. You can add new SFCs by following the same format.

