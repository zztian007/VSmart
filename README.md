# VSmart
VSmart (compiler Version identification for Smart contract) is a tool that takes in the bytecode of the smart contract to be analyzed and outputs the major compiler version used to produce it. The basic idea is to leverage deep neural networks to grasp version-indicative features from contracts’ normalized opcode sequences, and train classifiers on a dataset with ground-truth labels we collected from Etherscan. Details of our method are discussed in a paper that has been submited for peer-review.

To facilitate interested researchers conduct researches on this and relevant topics, we now made public the datasets we consturcted and VSmart’s source code.

The datasets consists of the collected raw information pages of the verified smart contracts on Etherscan, the wild smart contracts deployed on Ethereum blockchain, as well as our pre-processed files storing their bytecodes and labels. They are accessbile and can be downloaded from the GoogleDrive: https://drive.google.com/file/d/1q8-4b7iGlnWM_N4sOMpdHlg6Ga8lQ1pu/view?usp=sharing 

The source code of our implementation are accessbile on github: https://github.com/zztian007/VSmart 
