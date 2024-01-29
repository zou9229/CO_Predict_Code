# CO_Predict_Code
Use SAGE or other GNN model to predict CO emission level.
Core code and required data of the paper "Predicting Carbon Monoxide Emissions Level from Urabn Road Taxis Based on Graph Neural Network"
dgl_GNNS.py file for SAGE and other graph convolutional network model except GAT implementation code, clone down the project can be run directly after running, through the comment switch can choose to use what kind of graph convolutional network layer.
Due to the specificity of GAT, dgl_GAT.py and dgl_GAT2_official.py two files separately implement GAT and GAT2, also can be run directly.
Adjacency relation.csv file for the road network structure related information file, the specific use of the method can refer to the code in the dgl_GNNS.py file, where the comments are written clearly.
CGrade.csv file for the road carbon emissions related information files, specific methods of use can refer to the code in the dgl_GNNS.py file, where the comments are written clearly.
The semantic segmentation features of each road after processing.csv file for the semantic segmentation of the streetscape, and then all the street features of each street to do the average and then seek the results of the normalization, the specific use of the method can be referred to in the dgl_GNNS.py file in the code, where the comments are written very clearly. The comments are very clear.
The MLP_Predict.py file is a supplementary experimental model file for MLP to predict the carbon emission of streets, which can be run directly, and the code has been well commented.
If you need the original street view image data and cab GPS data, you can contact me. email: zou9229@qq.com
